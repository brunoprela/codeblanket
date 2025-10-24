/**
 * Multiple choice questions for Write-Ahead Log section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const writeaheadlogMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary reason Write-Ahead Logging (WAL) provides better performance than directly writing changes to data structures on disk?',
    options: [
      'WAL uses compression to reduce the size of data written',
      'WAL uses sequential writes which are much faster than random writes',
      'WAL stores data in memory instead of on disk',
      'WAL batches multiple operations into a single disk write',
    ],
    correctAnswer: 1,
    explanation:
      "WAL provides superior performance because it uses sequential writes (append-only) instead of random writes. On traditional spinning disks, sequential writes can be 10-1000x faster than random writes because the disk head doesn't need to move—it just continues writing at the current position. Even on SSDs, sequential writes are faster and cause less wear. In contrast, directly updating data structures (like B-trees) requires random disk seeks to different locations, causing disk head thrashing on HDDs. WAL appends all changes to a single log file sequentially, then updates in-memory structures (fast), and later flushes to disk asynchronously. Option 1 is incorrect—while some systems compress WAL, that's not the primary performance benefit. Option 3 is incorrect—WAL is on disk (for durability), not memory. Option 4 is true (group commit) but not the primary reason; sequential writes are the key.",
  },
  {
    id: 'mc2',
    question: 'In WAL-based recovery, what is the purpose of the "Undo" phase?',
    options: [
      'To replay all committed transactions from the log',
      'To identify which transactions were in progress at the time of crash',
      'To roll back changes made by uncommitted transactions',
      'To truncate the log file to save disk space',
    ],
    correctAnswer: 2,
    explanation:
      'The Undo phase rolls back changes made by uncommitted transactions to ensure atomicity (all-or-nothing property). When a system crashes, some transactions may have started but not committed—their changes were applied to in-memory structures (and possibly flushed to disk) but they never completed. During recovery, the Undo phase uses the "before images" stored in the WAL to reverse these partial changes, removing them from the database. This ensures only fully committed transactions survive the crash. Example: Transaction T starts, debits account A ($500), crashes before crediting account B. Undo phase reverses the $500 debit, restoring account A to original state. Option 1 describes the Redo phase (replay committed transactions). Option 2 describes the Analysis phase (identify transaction states). Option 4 is not part of the recovery process—log truncation happens after checkpointing.',
  },
  {
    id: 'mc3',
    question:
      'What problem does checkpointing solve in a Write-Ahead Logging system?',
    options: [
      'It prevents the database from running out of disk space',
      'It reduces recovery time by limiting how much of the log needs to be replayed',
      'It improves write performance by batching log entries',
      'It prevents unauthorized access to the log files',
    ],
    correctAnswer: 1,
    explanation:
      "Checkpointing solves the problem of unbounded recovery time. Without checkpoints, the WAL grows indefinitely, and recovery after a crash requires replaying the entire log from the beginning, which could take hours or days for a large, long-running system. Checkpointing periodically flushes all dirty (modified but not yet on disk) pages to disk and writes a checkpoint record to the log. This creates a known good state on disk. During recovery, the system only needs to replay log entries from the last checkpoint forward, not from the beginning. For example, if the log has 100GB of entries but the last checkpoint was at 90GB, recovery only processes 10GB. The system can also truncate log entries before the checkpoint since they're now reflected in the on-disk state. While checkpointing does help manage disk space (option 1), the primary purpose is fast recovery. Options 3 and 4 are unrelated to checkpointing's purpose.",
  },
  {
    id: 'mc4',
    question:
      'What is the trade-off between using consistent checkpoints versus fuzzy checkpoints in a WAL system?',
    options: [
      'Consistent checkpoints are faster but fuzzy checkpoints are more reliable',
      'Fuzzy checkpoints allow continued writes but have more complex recovery',
      'Consistent checkpoints use less disk space than fuzzy checkpoints',
      'Fuzzy checkpoints require more memory but are safer',
    ],
    correctAnswer: 1,
    explanation:
      "Fuzzy checkpoints allow writes to continue during checkpointing (no downtime), but recovery is slightly more complex. With consistent checkpoints, all writes are stopped, all dirty pages are flushed to disk, a checkpoint record is written, and then writes resume. This is simple (recovery just starts from checkpoint) but causes downtime (seconds to minutes). With fuzzy checkpoints, writes continue during the checkpoint process. The checkpoint records a starting LSN (log sequence number), dirty pages are flushed asynchronously, and writes can happen concurrently. During recovery, the system must replay from the checkpoint's recorded LSN (not just the checkpoint itself) because some writes may have occurred during checkpointing. This adds slight complexity to recovery logic but eliminates downtime, making it essential for online production systems. PostgreSQL, MySQL, and other production databases use fuzzy checkpoints. Option 1 is backward (fuzzy has no downtime but more complex recovery). Options 3 and 4 are incorrect—memory and disk space usage are similar.",
  },
  {
    id: 'mc5',
    question:
      "In a group commit optimization, what is the main benefit of batching multiple transactions' log writes together?",
    options: [
      'It reduces the total size of the log file through compression',
      'It reduces the number of expensive fsync operations needed',
      'It allows transactions to execute in parallel on multiple CPU cores',
      'It eliminates the need for checkpointing',
    ],
    correctAnswer: 1,
    explanation:
      "Group commit reduces the number of expensive fsync (force sync to disk) operations by batching multiple transactions' log entries and flushing them together in a single fsync. Fsync is slow (typically 10-100ms) because it must wait for data to physically reach persistent storage. Without group commit, each transaction requires its own fsync, limiting throughput to ~10-100 transactions/second. With group commit, if 10 transactions arrive within a short window (e.g., 10ms), their log entries are written together and a single fsync flushes all 10. This reduces average latency per transaction and dramatically increases throughput (potentially 1000+ transactions/second). Example: Without group commit: T1 fsync (10ms), T2 fsync (10ms), T3 fsync (10ms) = 30ms total, 3 fsyncs. With group commit: T1, T2, T3 all write to buffer, single fsync (10ms) = 10ms total, 1 fsync. Option 1 is incorrect—group commit doesn't compress. Option 3 is incorrect—it's about I/O efficiency, not CPU parallelism. Option 4 is incorrect—checkpointing is still needed.",
  },
];
