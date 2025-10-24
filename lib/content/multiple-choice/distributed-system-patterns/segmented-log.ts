/**
 * Multiple choice questions for Segmented Log section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const segmentedlogMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary advantage of using segmented logs over a single large log file when implementing log compaction?',
    options: [
      'Segmented logs use less total disk space',
      'Individual closed segments can be compacted without affecting active writes',
      'Segmented logs automatically compress data',
      'Reading from segmented logs is always faster than from a single file',
    ],
    correctAnswer: 1,
    explanation:
      "The primary advantage for compaction is that individual closed segments can be compacted independently without blocking active writes. With a single large log, compacting (removing duplicates, keeping latest values) requires rewriting the entire file, which is difficult to do while new writes are coming in—you'd need complex coordination or temporary write blocking. With segmented logs, closed (immutable) segments can be compacted in the background by dedicated threads while the active segment continues accepting writes normally. For example, Cassandra compacts old SSTables independently, Kafka compacts closed log segments, all while new data is written to the active segment. This enables efficient resource management and improved read performance (fewer segments to check after compaction) without impacting write availability. Option 1 is incorrect—total space is similar. Option 3 is incorrect—compression is separate from segmentation. Option 4 is incorrect—read speed depends on many factors, not just segmentation.",
  },
  {
    id: 'mc2',
    question: 'In Apache Kafka, how are segment files named?',
    options: [
      'Sequential numbers starting from 1 (segment-001.log, segment-002.log)',
      'The starting offset of the segment (00000000001000000000.log for offset 1,000,000,000)',
      'Timestamps when the segment was created (20240115-103045.log)',
      'Random UUIDs to ensure uniqueness (a1b2c3d4-e5f6-7890.log)',
    ],
    correctAnswer: 1,
    explanation:
      "Kafka names segment files with the starting offset of the segment, zero-padded to 20 digits. For example, 00000000001000000000.log represents the segment starting at offset 1,000,000,000. This naming scheme provides several benefits: (1) Given an offset, you can quickly determine which segment file contains it by finding the segment with the largest starting offset ≤ target offset. (2) Segments are naturally sorted by name, making it easy to iterate through them in order. (3) The names are deterministic and meaningful for operators debugging issues. Each segment also has associated .index and .timeindex files with the same base name. Example: For partition topic-0, you might see 00000000000000000000.log (first segment), 00000000000001000000.log (second segment starting at offset 1,000,000), etc. Option 1 is incorrect—sequential numbers don't convey offset information. Option 3 is incorrect—timestamps aren't used in file names. Option 4 is incorrect—UUIDs would make finding the right segment harder.",
  },
  {
    id: 'mc3',
    question:
      'Which of the following is NOT a valid trigger for rotating to a new segment in a production system?',
    options: [
      'The current segment reaches a size limit (e.g., 1GB)',
      'A fixed time interval has elapsed (e.g., 1 hour)',
      'The segment contains a specific number of records (e.g., 10 million)',
      'A specific key is written to the log',
    ],
    correctAnswer: 3,
    explanation:
      'Rotating based on a specific key being written is not a valid trigger—segmentation should be based on uniform, predictable criteria, not the content of individual writes. Standard triggers are: (1) Size-based: When segment reaches a configured size limit (e.g., 1GB), prevents segments from growing too large and ensures manageable file sizes. (2) Time-based: After a fixed interval (e.g., 1 hour), aligns segments with retention policies and handles low-throughput scenarios where size limits might never be reached. (3) Record-count-based: After N records, useful when records are uniform size and you want predictable segment sizes. Production systems typically use a combination: "whichever comes first." Example: Kafka segment.ms (time) and segment.bytes (size)—if writes are fast, size limit triggers rotation; if slow, time limit triggers it. This handles both high and low throughput. Content-based rotation (option 4) would be unpredictable and could create highly variable segment sizes.',
  },
  {
    id: 'mc4',
    question:
      'What is the purpose of index files (e.g., .index) that accompany segment files in systems like Kafka?',
    options: [
      'To store compressed versions of the data',
      'To enable fast lookup of specific offsets without scanning the entire segment',
      'To maintain checksums for detecting corruption',
      'To track which consumers have read the segment',
    ],
    correctAnswer: 1,
    explanation:
      "Index files enable fast lookup of specific offsets within a segment without scanning the entire segment file. They map offsets (or keys) to physical byte positions in the segment file. For example, Kafka's .index file is a sparse index mapping offsets to byte positions every 4KB or so. To read offset 1,500,000, Kafka: (1) Opens segment 00000000001000000000.log (starts at offset 1M, so contains 1.5M). (2) Looks in 00000000001000000000.index to find the index entry ≤ 1,500,000 (e.g., offset 1,499,900 → position 3,000,000 bytes). (3) Seeks to position 3,000,000 in the segment file. (4) Scans forward sequentially to find exact offset 1,500,000 (fast because only scanning ~100 offsets). Without the index, Kafka would need to scan the entire segment from the beginning (potentially gigabytes, taking seconds). With the index, lookup is near-instant. Option 1 is incorrect—compression is separate. Option 3 is incorrect—checksums are typically in the data itself, not separate index files. Option 4 is incorrect—consumer positions are tracked elsewhere (in Kafka, in __consumer_offsets topic).",
  },
  {
    id: 'mc5',
    question:
      'In Kafka log compaction, what happens to a record with a null value?',
    options: [
      'It is skipped and ignored during compaction',
      'It causes an error and prevents compaction from completing',
      'It is treated as a tombstone marking the key for deletion',
      'It is stored as-is like any other record',
    ],
    correctAnswer: 2,
    explanation:
      'In Kafka log compaction, a record with a null value is treated as a tombstone—a marker indicating that the key should be deleted. When compaction runs, tombstones are used to delete all previous values for that key from the log. However, tombstones themselves are retained temporarily (controlled by delete.retention.ms, default 24 hours) to ensure all consumers see the deletion before the tombstone is removed. After the retention period, the tombstone and all records for that key are fully removed from the log. Example: key="user:123" had values v1, v2, v3. Producer sends key="user:123", value=null (tombstone). After compaction: All previous values (v1, v2, v3) removed. Tombstone retained for 24 hours (so consumers can see deletion). After 24 hours: Tombstone also removed. Key "user:123" no longer exists in log. This enables Kafka to be used as a changelog or database replica, where deletes are properly propagated. Option 1 is incorrect—tombstones are explicitly processed. Option 2 is incorrect—null values are valid and expected. Option 4 is incorrect—tombstones have special deletion semantics.',
  },
];
