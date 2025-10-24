/**
 * Write-Ahead Log (WAL) Section
 */

export const writeaheadlogSection = {
  id: 'write-ahead-log',
  title: 'Write-Ahead Log (WAL)',
  content: `The Write-Ahead Log (WAL) is a fundamental technique in database systems and distributed systems that ensures **durability** and **atomicity** of operations. It's the backbone of crash recovery in most production databases.

## What is a Write-Ahead Log?

A Write-Ahead Log is a **sequential log file** where all modifications are written **before** they are applied to the actual data structures or database.

**Core Principle**: "Log first, modify later"

Before making any change to the data:
1. **Write** the intended change to the log (on disk)
2. **Flush** the log to durable storage
3. **Apply** the change to in-memory data structures
4. Eventually **persist** in-memory changes to disk

**Key Guarantee**: If a change is logged, it will eventually be applied (even after crashes).

---

## Why Write-Ahead Logging?

### **1. Durability (ACID Property)**

Without WAL:
\`\`\`
1. Modify in-memory data structure
2. System crashes before writing to disk
3. Change is lost forever ❌
\`\`\`

With WAL:
\`\`\`
1. Write change to log (flushed to disk)
2. System crashes
3. On recovery: replay log, reapply change ✅
\`\`\`

**Result**: Once a client receives "committed", the operation **survives crashes**.

### **2. Atomicity (All-or-Nothing)**

Complex operations involve multiple steps:
\`\`\`
Transfer $100 from Account A to Account B:
  1. Deduct $100 from A
  2. Add $100 to B
\`\`\`

If crash happens between steps 1 and 2:
- Without WAL: A is debited but B is not credited (data inconsistency)
- With WAL: Both operations are logged atomically, replay ensures both happen

### **3. Fast Writes**

**Sequential writes** to log are **much faster** than **random writes** to data structures:
- Log: Sequential append (disk head doesn't move)
- Data structures: Random updates (disk head moves constantly)

**Performance**: Sequential writes can be 10-1000x faster than random writes on spinning disks.

### **4. Crash Recovery**

When system crashes:
1. Read WAL from last checkpoint
2. Replay all logged operations
3. Restore system to consistent state
4. Resume normal operation

**Recovery time** depends on log size and checkpoint frequency.

---

## How WAL Works

### **Write Path**

\`\`\`
Client: INSERT INTO users VALUES (1, 'Alice')

1. Lock: Acquire lock on page containing users table
2. Log Entry: Create log record:
   {
     LSN: 1001,                    // Log Sequence Number
     Type: INSERT,
     Table: users,
     Data: {id: 1, name: 'Alice'}
   }
3. Write Log: Append to WAL buffer
4. Flush Log: Force WAL to disk (fsync)
5. Modify Memory: Update in-memory page
6. Acknowledge: Return success to client
7. Later: Flush dirty page to disk (asynchronous)
\`\`\`

**Critical Order**: Log must hit disk **before** acknowledging client.

### **Log Entry Structure**

Each log entry typically contains:

\`\`\`
{
  LSN: 1001,                    // Unique Log Sequence Number
  PrevLSN: 1000,                // Previous LSN (linked list)
  TransactionID: 42,            // Which transaction
  Type: INSERT | UPDATE | DELETE | BEGIN | COMMIT,
  PageID: 123,                  // Which page is affected
  Before: {...},                // Old value (for undo)
  After: {...},                 // New value (for redo)
  Timestamp: 2024-01-15T10:30:00
}
\`\`\`

**LSN (Log Sequence Number)**: Monotonically increasing, unique identifier for each log entry.

### **Recovery Process**

When system crashes and restarts:

**Phase 1: Analysis**
- Scan WAL to identify transactions
- Determine which transactions were committed
- Identify dirty pages (pages modified but not flushed)

**Phase 2: Redo**
- Replay all committed transactions from WAL
- Reapply changes to data structures
- Ensures durability

**Phase 3: Undo**
- Roll back uncommitted transactions
- Uses "before" images from log
- Ensures atomicity

**Example**:
\`\`\`
Log:
  LSN 1000: BEGIN Transaction T1
  LSN 1001: UPDATE Account A, deduct $100
  LSN 1002: UPDATE Account B, add $100
  LSN 1003: COMMIT Transaction T1
  LSN 1004: BEGIN Transaction T2
  LSN 1005: UPDATE Account C, deduct $50
  [CRASH]

Recovery:
  - Analysis: T1 committed, T2 not committed
  - Redo: Replay LSN 1001, 1002 (in case not flushed)
  - Undo: Roll back LSN 1005 (T2 not committed)
\`\`\`

---

## Checkpointing

Problem: WAL grows indefinitely, recovery takes longer.

**Solution**: Periodic **checkpoints** - snapshots of the database state.

### **How Checkpointing Works**

1. **Trigger**: After N operations or T seconds
2. **Flush**: Write all dirty pages to disk
3. **Checkpoint Record**: Write checkpoint marker to WAL
   - Includes list of active transactions
   - Includes LSN of checkpoint
4. **Truncate**: Old log entries (before checkpoint) can be deleted

**Recovery After Checkpoint**:
- Start from last checkpoint (not beginning of log)
- Only replay operations after checkpoint
- Much faster recovery

**Example**:
\`\`\`
Time 0:  Database state = S0
Time 1:  Operations O1, O2, O3 (logged)
Time 2:  Checkpoint C1 created
         - Flush dirty pages (S0 + O1, O2, O3 = S1)
         - Write checkpoint record
Time 3:  Operations O4, O5 (logged)
[CRASH]

Recovery:
  - Start from checkpoint C1 (state S1)
  - Replay only O4, O5
  - No need to replay O1, O2, O3
\`\`\`

### **Types of Checkpoints**

**1. Consistent Checkpoint**
- Stop all writes
- Flush all dirty pages
- Write checkpoint
- Resume writes

**Pros**: Simple, guarantees consistency
**Cons**: Downtime during checkpoint

**2. Fuzzy Checkpoint**
- Allow writes to continue during checkpoint
- Record LSN at checkpoint start
- Flush dirty pages asynchronously
- Recovery replays from recorded LSN

**Pros**: No downtime
**Cons**: Slightly more complex recovery

**Production systems** use fuzzy checkpoints to avoid downtime.

---

## WAL Optimization Techniques

### **1. Group Commit**

**Problem**: Flushing log to disk for every single transaction is expensive (fsync is slow).

**Solution**: Batch multiple transactions' log entries and flush together.

\`\`\`
Without Group Commit:
  Transaction T1: Write log, fsync (10ms)
  Transaction T2: Write log, fsync (10ms)
  Transaction T3: Write log, fsync (10ms)
  Total: 30ms, 3 fsyncs

With Group Commit:
  Transaction T1: Write log
  Transaction T2: Write log
  Transaction T3: Write log
  All together: fsync (10ms)
  Total: 10ms, 1 fsync
\`\`\`

**Trade-off**: Slightly higher latency for individual transactions (wait for batch), but much higher throughput.

**Configuration**: 
- Batch size: 10-100 transactions
- Timeout: 10-100ms (don't wait forever)

### **2. Log Buffer**

**Problem**: Writing to disk for every log entry is slow.

**Solution**: Use in-memory log buffer, flush periodically or when full.

\`\`\`
1. Append log entry to memory buffer (fast)
2. When buffer full or timeout: Flush to disk
3. Continue appending to buffer
\`\`\`

**Trade-off**: Risk of losing buffered entries if crash before flush (configurable durability).

**PostgreSQL**: \`commit_delay\` and \`commit_siblings\` parameters control this.

### **3. Parallel WAL Writes**

For high throughput:
- Multiple WAL writers writing to different segments
- Merge segments during recovery

**Used by**: High-performance databases like CockroachDB

### **4. Log Compression**

**Problem**: Log takes significant disk space.

**Solution**: Compress log entries before writing.
- Reduce disk I/O
- Faster replication (less data to transfer)
- Trade-off: CPU cost of compression/decompression

---

## WAL in Real-World Systems

### **PostgreSQL**

PostgreSQL's WAL is called **pg_wal** (formerly pg_xlog).

**Configuration**:
\`\`\`
wal_level = replica           # Amount of information written
max_wal_size = 1GB            # Checkpoint triggered when exceeded
wal_buffers = 16MB            # In-memory buffer size
fsync = on                    # Force writes to disk
synchronous_commit = on       # Wait for fsync before commit
\`\`\`

**WAL Segments**: 16MB files, rotated as filled
- Files: 000000010000000000000001, 000000010000000000000002, ...

**Point-in-Time Recovery (PITR)**:
- Archive WAL segments
- Restore from base backup
- Replay WAL to specific timestamp

**Replication**:
- Primary streams WAL to replicas
- Replicas replay WAL to stay in sync

### **MySQL (InnoDB)**

InnoDB uses redo log (WAL) and undo log.

**Configuration**:
\`\`\`
innodb_log_file_size = 512MB    # Size of each log file
innodb_log_files_in_group = 2   # Number of log files (circular)
innodb_log_buffer_size = 16MB   # In-memory log buffer
innodb_flush_log_at_trx_commit = 1  # Durability setting
\`\`\`

**Durability Settings**:
- \`= 0\`: Flush every second (fast, risk of loss)
- \`= 1\`: Flush on every commit (slow, durable)
- \`= 2\`: Write to OS cache, let OS flush (middle ground)

**Double Write Buffer**: Additional safety mechanism to prevent partial page writes.

### **Apache Kafka**

Kafka's log is effectively a distributed WAL.

**Concept**: Messages are appended to log, never updated.

**Properties**:
- Sequential writes (high throughput)
- Retention policy (time or size-based)
- Compaction for key-based deduplication
- Replication across brokers

**Use Cases**:
- Message queue
- Event sourcing
- Change data capture (CDC)

**Performance**: Can handle millions of writes per second.

### **Redis (AOF - Append Only File)**

Redis can use WAL-like persistence called AOF.

**How it works**:
1. Every write operation is logged to AOF
2. On restart, replay AOF to reconstruct state
3. AOF can be rewritten (compaction) to reduce size

**Configuration**:
\`\`\`
appendonly yes
appendfsync everysec    # always | everysec | no
\`\`\`

**Trade-off**: Durability vs performance
- \`always\`: Durable, slowest
- \`everysec\`: Good balance (default)
- \`no\`: Fast, risk of data loss

### **RocksDB (Used by CockroachDB, TiDB)**

RocksDB implements WAL for durability.

**LSM Tree Architecture**:
1. Writes go to WAL
2. Writes go to memtable (in-memory)
3. Memtable flushed to SSTable (disk) periodically
4. SSTables compacted over time

**WAL is crucial** because memtable is in memory (lost on crash).

---

## WAL vs Other Durability Techniques

### **WAL vs Shadow Paging**

**Shadow Paging**:
- Make copy of page before modifying
- Modify copy
- Atomically switch pointer to new page

**Comparison**:
\`\`\`
WAL:
  + Sequential writes (fast)
  + Compact log
  - Complex recovery

Shadow Paging:
  + Simpler recovery
  - Random writes (slow)
  - Disk space overhead (two copies)
\`\`\`

**Modern systems prefer WAL** due to performance.

### **WAL vs Snapshotting**

**Snapshotting**: Periodically save entire state to disk.

**Comparison**:
\`\`\`
WAL:
  + Fast writes (log only)
  + Granular recovery
  - Log grows over time

Snapshotting:
  + Full state saved
  - Slow writes (entire state)
  - Long recovery if no recent snapshot
\`\`\`

**Best approach**: Combine both (WAL + periodic snapshots/checkpoints).

---

## Advanced Topics

### **Log Structured Merge Trees (LSM)**

WAL is a core component of LSM trees:

\`\`\`
1. Write → WAL (durability)
2. Write → Memtable (fast in-memory structure)
3. Memtable full → Flush to SSTable (sorted, immutable)
4. SSTables compacted → Merge and deduplicate
\`\`\`

**Used by**: RocksDB, LevelDB, Cassandra, HBase, ScyllaDB

**Benefits**: 
- High write throughput
- Good for write-heavy workloads

### **Distributed WAL**

In distributed systems, WAL is replicated across nodes.

**Example: Raft Log**
- Log entries replicated to majority of nodes
- Once majority acknowledges, entry is committed
- Each node has its own copy of log

**Consensus**: Ensures all nodes agree on log order.

### **Event Sourcing**

**Concept**: Store all changes as events (never update in place).

**Relationship to WAL**: Event log IS the database.

\`\`\`
Traditional:
  UPDATE accounts SET balance = balance - 100 WHERE id = 1

Event Sourcing:
  Event 1: AccountDebitedEvent {accountId: 1, amount: 100, timestamp: ...}
  Event 2: AccountCreditedEvent {accountId: 2, amount: 100, timestamp: ...}
\`\`\`

**Benefits**:
- Complete audit trail
- Time travel (replay to any point)
- Multiple views from same events (CQRS)

**Challenges**:
- Query complexity (must replay events)
- Storage growth
- Schema evolution

---

## Implementation Considerations

### **Disk Performance**

**SSDs vs HDDs**:
- HDDs: Sequential writes much faster than random (WAL helps significantly)
- SSDs: Random writes faster, but sequential still better
- NVMe: Very fast, but WAL still helps with durability and crash recovery

**Fsync Behavior**:
- \`fsync()\`: Expensive system call (forces OS to flush to disk)
- Without fsync: Data may sit in OS cache (lost on crash)
- Battery-backed write cache: Can acknowledge without fsync (hardware durability)

### **Disk Space Management**

**Log Rotation**:
\`\`\`
1. Log reaches size limit (e.g., 1GB)
2. Close current log file
3. Open new log file
4. Old log files archived or deleted (after checkpoint)
\`\`\`

**Archival**: 
- Keep old logs for point-in-time recovery
- Compress archived logs
- Move to cheaper storage (S3, Glacier)

### **Monitoring**

**Key Metrics**:

1. **WAL Growth Rate**: Bytes/second written to log
   - High growth: Many writes
   - Sudden increase: Investigate workload

2. **WAL Disk Usage**: Current size of WAL
   - Approaching limit: Trigger checkpoint
   - Not recycling: Checkpoint issues

3. **Fsync Latency**: Time to flush log to disk
   - P99 latency > 100ms: Disk contention
   - Spikes: Storage issues

4. **Checkpoint Frequency**: Checkpoints per hour
   - Too frequent: Too much overhead
   - Too rare: Slow recovery, large logs

5. **Recovery Time**: Time to recover from crash
   - Slow recovery: Reduce checkpoint interval

**Alerts**:
- WAL disk 80% full
- Fsync P99 > 100ms
- Checkpoint taking > 30s

---

## Common Pitfalls

### **1. Not Flushing Log Before Acknowledging**

**Problem**:
\`\`\`
1. Write to log buffer
2. Return success to client
3. Crash before log flushed to disk
4. Change is lost
\`\`\`

**Solution**: Always fsync before acknowledge (unless explicitly configuring for performance over durability).

### **2. Ignoring Fsync Failures**

**Problem**: Fsync can fail (disk full, hardware error).

**Solution**: Check return value, handle errors (alert, retry, fail-safe).

### **3. Log and Data on Same Disk**

**Problem**: Disk failure loses both log and data.

**Solution**: 
- Put WAL on separate disk (ideally separate physical device)
- Replicate WAL to remote nodes

### **4. Not Testing Recovery**

**Problem**: Recovery code rarely executed, may be buggy.

**Solution**: 
- Regularly test crash recovery (chaos engineering)
- Automated tests that simulate crashes
- Verify recovery correctness

### **5. Unbounded Log Growth**

**Problem**: No checkpoints, log grows indefinitely.

**Solution**: Implement automatic checkpointing (size or time-based).

---

## Interview Tips

### **Key Concepts to Explain**

1. **Why WAL**: Durability, atomicity, performance (sequential writes)
2. **How it works**: Log before apply, recovery by replay
3. **Checkpointing**: Limit recovery time, truncate log
4. **Trade-offs**: Durability vs performance (fsync cost)

### **Common Interview Questions**

**Q: Why is WAL faster than directly updating data on disk?**
A: "WAL uses sequential writes (append-only log), which are 10-1000x faster than random writes on spinning disks. The log is also typically smaller than the entire dataset. Data structures are updated in memory (fast) and flushed asynchronously later."

**Q: What happens if the system crashes right after writing to the log but before applying the change?**
A: "During recovery, the system reads the WAL and replays all committed operations. The change is reapplied to the data structures, ensuring durability. This is the 'redo' phase of recovery."

**Q: How do you prevent the WAL from growing too large?**
A: "Use checkpointing: periodically flush all dirty pages to disk and write a checkpoint record. After a checkpoint, log entries before it can be safely deleted since they're reflected in the on-disk state. Recovery starts from the last checkpoint, not the beginning."

**Q: What's the trade-off between durability and performance?**
A: "Strict durability requires fsync on every commit (wait for disk I/O), which is slow (10-100ms). Performance can be improved by: (1) Group commit (batch fsyncs), (2) Asynchronous flush (risk losing last few seconds of data), (3) Battery-backed cache (hardware durability). Choice depends on application requirements."

---

## Summary

Write-Ahead Logging is fundamental to data durability and crash recovery:

1. **Core Idea**: Log changes before applying them
2. **Guarantees**: Durability (changes survive crashes), Atomicity (all-or-nothing)
3. **Performance**: Sequential writes are fast
4. **Recovery**: Replay log to restore state
5. **Checkpointing**: Limit log growth and recovery time
6. **Real-World**: PostgreSQL, MySQL, Kafka, Redis, RocksDB all use WAL
7. **Trade-offs**: Durability vs performance, log size vs recovery time

**Interview Focus**: Understand the problem WAL solves (crash recovery), how it works (log then apply), and trade-offs (fsync cost, log growth).
`,
};
