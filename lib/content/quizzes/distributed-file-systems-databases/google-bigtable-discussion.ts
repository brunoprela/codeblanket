/**
 * Quiz questions for Google BigTable section
 */

export const bigtableQuiz = [
  {
    id: 'q1',
    question:
      'Explain the LSM-tree write path in BigTable. Why does this design enable fast writes?',
    sampleAnswer:
      "BigTable uses LSM-tree (Log-Structured Merge-tree) for writes: (1) Write to commit log (WAL) in GFS - sequential append, very fast. (2) Insert into memtable (in-memory sorted buffer) - in-memory write, instant. (3) Return success to client immediately. (4) Later: When memtable full (~64 MB), flush to immutable SSTable on GFS. (5) Eventually: Compact SSTables to merge and remove old versions. Why fast: (1) No random disk writes! All disk writes are sequential (commit log, SSTable flush). Random writes are 100x slower than sequential. (2) Batching - memtable accumulates writes, flush in bulk. (3) In-memory path - writes served from RAM (nanoseconds). (4) Asynchronous persistence - SSTable creation doesn't block client. Trade-offs: (1) Reads check multiple places (memtable + multiple SSTables). (2) Compaction overhead (background I/O). (3) Eventually consistent reads (until compaction). But for write-heavy workloads (like Google\'s logs, MapReduce output), this is ideal. LSM-tree is fundamental to modern NoSQL (HBase, Cassandra, RocksDB).",
    keyPoints: [
      'Write to commit log (sequential, fast) + memtable (in-memory)',
      'No random disk writes - all sequential',
      "Asynchronous SSTable flush (doesn't block client)",
      'Trade-off: Complex reads (multiple SSTables)',
      'Optimized for write-heavy workloads',
    ],
  },
  {
    id: 'q2',
    question:
      'How do Bloom filters improve BigTable read performance? What are the trade-offs?',
    sampleAnswer:
      "Bloom filters are probabilistic data structures that quickly answer 'is key X in SSTable Y?' Each SSTable has Bloom filter (~10 KB in memory). On read: Check Bloom filter first. If 'definitely not present' → skip SSTable (avoid disk I/O). If 'might be present' → read SSTable from disk. False positive rate ~1% (configurable). Benefits: (1) Avoid 99% of unnecessary disk reads. Example: Reading key from table with 100 SSTables. Without Bloom: Read all 100 SSTables (100 disk I/Os). With Bloom: Check 100 Bloom filters (in memory), read only SSTables that might contain key (~1-2 disk I/Os). (2) Space efficient - 10 KB filter for GB-sized SSTable. (3) Fast - O(1) hash lookups. Trade-offs: (1) False positives - 1% of time, read SSTable unnecessarily. (2) Memory overhead - must keep all Bloom filters in memory. (3) Only works for point lookups (not range scans). Despite false positives, net benefit is huge. Bloom filters are critical for BigTable/HBase/Cassandra performance.",
    keyPoints: [
      'Bloom filter: Quickly check if key might be in SSTable',
      'Avoid 99% of unnecessary disk reads',
      '10 KB in memory per SSTable (space efficient)',
      'Trade-off: 1% false positive rate',
      'Critical optimization for read-heavy workloads',
    ],
  },
  {
    id: 'q3',
    question:
      'Why does BigTable use tablets instead of storing entire table on one server? How does the tablet hierarchy enable scalability?',
    sampleAnswer:
      "Tablets = contiguous ranges of rows. Table split into tablets (default ~100-200 MB each), distributed across tablet servers. Why: (1) Distribute load - hot rows go to different servers. One server can't handle entire table. (2) Parallel operations - multiple servers serve different tablets simultaneously. (3) Incremental scaling - add servers, redistribute tablets (no downtime). (4) Fault tolerance - tablet server fails, master reassigns tablets to other servers. 3-level hierarchy: Chubby → ROOT tablet → META tablets → User tablets. ROOT tablet (stored in Chubby): Maps to META tablets. META tablets: Map (tablet location) → (user tablet). User tablets: Actual data. Each META tablet can address ~128K user tablets. Total: 128K * 128K = 2^34 tablets = enormous scale! Client lookup: (1) Read ROOT from Chubby (cached). (2) Read META tablet (cached). (3) Contact tablet server directly. Future reads: Use cached location (0 network hops). Tablets enable BigTable to scale to billions of rows and thousands of servers without bottlenecks.",
    keyPoints: [
      'Tablets = ranges of rows, unit of distribution',
      'Distribute load across many tablet servers',
      '3-level hierarchy: ROOT → META → User tablets',
      'Can address 2^34 tablets (massive scale)',
      'Client caches tablet locations (fast reads)',
    ],
  },
];
