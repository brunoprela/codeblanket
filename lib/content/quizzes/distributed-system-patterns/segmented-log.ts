/**
 * Quiz questions for Segmented Log section
 */

export const segmentedlogQuiz = [
  {
    id: 'q1',
    question:
      'Explain why segmented logs are necessary and describe three key problems they solve compared to using a single large log file.',
    sampleAnswer: `Segmented logs break a potentially infinite log into multiple fixed-size segments, solving critical operational problems. Problem 1: Deletion/Cleanup. Single large log: To delete old data, must rewrite entire file (expensive, time-consuming). With segments: Simply delete entire old segment files. Example—Kafka retaining 7 days: delete segments older than 7 days in seconds, no rewrite needed. Problem 2: Compaction. Single log: Compacting (removing duplicates, keeping latest values) requires rewriting entire log while new writes continue. With segments: Compact individual closed segments without affecting active segment. Cassandra compacts SSTables independently, allowing concurrent compaction and writes. Problem 3: Recovery. Single 100GB log: Sequential read during recovery (single thread, slow). With 10×10GB segments: Parallel recovery—read multiple segments concurrently with multiple threads, 10x faster. Each segment is immutable once closed, enabling safe concurrent access. Additional benefits: (1) Easier archival—move cold segments to cheaper storage (S3, Glacier). (2) Better I/O performance—different caching policies per segment (hot in memory, cold on disk). (3) Bounded impact—corruption affects one segment, not entire log. Production systems (Kafka, Cassandra, RocksDB) all use segmented logs because these operations are fundamental to maintaining large-scale, long-running systems.`,
    keyPoints: [
      'Deletion: Delete entire segment files vs rewriting single large file (Kafka 7-day retention)',
      'Compaction: Compact individual closed segments independently without blocking writes',
      'Recovery: Parallel recovery across segments (10x faster) vs sequential single file',
      'Archival: Move old segments to cheaper storage (S3), easier tiering',
      'Real-world: Kafka, Cassandra, RocksDB all use segmented logs for operational efficiency',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare log compaction in Kafka with SSTable compaction in Cassandra. What are the similarities and differences in their approaches?',
    sampleAnswer: `Both use compaction to remove redundant data from segmented logs, but with different goals and mechanisms. Kafka Log Compaction: Goal—retain only latest value for each key, enabling infinite log retention bounded by number of unique keys. Process: Scan segments, identify duplicates for same key, keep latest based on offset. Deleted keys marked with null (tombstone), removed after grace period. Use case: Change Data Capture (CDC), database snapshots. Example: User record updates—keep only latest version per user. Log bounded by user count, not time. Cassandra SSTable Compaction: Goal—merge SSTables to improve read performance and reclaim space. Multiple strategies: (1) Size-Tiered: Merge similar-sized SSTables. (2) Leveled: Organize into levels like LSM tree. (3) Time-Window: Group by time, ideal for time-series data. Process: Read multiple SSTables, merge overlapping key ranges, write new SSTable, delete old ones. Similarities: (1) Both operate on immutable segments. (2) Both reclaim space by removing redundant data. (3) Both can be resource-intensive (I/O, CPU). (4) Both use tombstones for deletions with grace periods. Differences: Kafka focuses on key-based deduplication (latest value). Cassandra focuses on merging and reorganizing for read performance. Kafka compaction optional (clean up policy). Cassandra compaction mandatory (LSM tree requirement). When to use: Kafka compaction for event streams needing latest state per entity. Cassandra compaction (always running) to maintain read performance in LSM architecture.`,
    keyPoints: [
      'Kafka compaction: Retain latest value per key, infinite retention bounded by unique keys',
      'Cassandra compaction: Merge SSTables for read performance, multiple strategies (STCS, LCS, TWCS)',
      'Similarities: Immutable segments, remove redundancy, tombstones, resource-intensive',
      'Differences: Kafka key-deduplication optional, Cassandra merge for performance mandatory',
      'Use cases: Kafka for CDC/state snapshots, Cassandra for LSM read optimization',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a segment rotation policy for a message queue system. What triggers would you use and how would you balance segment size, rotation frequency, and retention?',
    sampleAnswer: `Effective segment rotation requires multiple triggers and careful trade-off balancing. Triggers (use whichever hits first): (1) Size-based: Rotate when segment reaches 1GB. Rationale: Manageable file size for individual operations (compaction, transfer, deletion), but large enough to avoid excessive files. Too small (10MB) = too many files (file system overhead). Too large (100GB) = slow operations. (2) Time-based: Rotate every 1 hour. Rationale: Aligns with retention policy (e.g., "keep 24 hours" = 24 segments, easy deletion). Handles both high and low throughput—high throughput hits size limit first, low throughput hits time limit. (3) Record-count-based: Rotate after 10 million messages. Rationale: Predictable segment sizes for uniform message sizes. Easier capacity planning. Combined approach example for message queue: Segment size: 1GB; Time limit: 1 hour; Record limit: 10M messages. High throughput (2GB/hour): Size trigger at 30 minutes. Low throughput (100MB/hour): Time trigger at 1 hour. Spiky traffic: Record limit acts as safety valve. Retention policy: Keep 7 days of data = 168 hour-long segments max (if time-triggered). Monitor total size (e.g., cap at 1TB), delete oldest if exceeded. Special considerations: (1) Peak traffic: Ensure rotation happens frequently enough that spikes don't create huge segments. (2) Compaction: If using compaction, rotate small enough that compaction completes reasonably. (3) Replication: Smaller segments replicate faster.Typical production: 100MB- 1GB segments, 1 - hour max age, size and time triggers combined.`,
    keyPoints: [
      'Multiple triggers: Size (1GB), time (1 hour), records (10M)—whichever hits first',
      'Size considerations: Too small = file system overhead, too large = slow operations',
      'Time alignment: Match retention policy (hourly segments for daily retention)',
      'Handle variable throughput: High throughput hits size first, low hits time',
      'Retention: Time-based (7 days) or size-based (1TB), delete oldest segments',
    ],
  },
];
