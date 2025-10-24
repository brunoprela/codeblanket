/**
 * Quiz questions for Write-Ahead Log section
 */

export const writeaheadlogQuiz = [
    {
        id: 'q1',
        question:
            'Explain how Write-Ahead Logging (WAL) ensures durability and atomicity in database systems, and why it performs better than directly writing to data structures on disk.',
        sampleAnswer: `WAL ensures durability and atomicity through the "log first, modify later" principle. Before any data modification, the change is written to a sequential log file and flushed to disk (fsync). Only then is the in-memory data structure updated. Durability: Once a client receives "committed," the operation is in the WAL on disk. Even if the system crashes before the in-memory changes are flushed, recovery replays the WAL to restore the operation. Atomicity: Complex operations (e.g., transfer $100 from A to B) are logged atomically. If crash occurs mid-operation, recovery either replays both steps (debit A, credit B) or neither, ensuring all-or-nothing semantics. Performance advantage: WAL uses sequential writes (append-only), which are 10-1000x faster than random writes on spinning disks. Disk head doesn't move, maximizing throughput.In contrast, directly updating data structures requires random writes(seek to different locations), causing disk head thrashing.Example: PostgreSQL can handle 10,000 writes / sec with WAL(sequential) vs 100 writes / sec with direct random writes.Even on SSDs, sequential writes are faster and wear more evenly.Additionally, in-memory updates are fast—WAL provides durability without sacrificing write performance.`,
        keyPoints: [
            'Log first, modify later: Write to WAL and flush before applying to data structures',
            'Durability: WAL on disk survives crashes, recovery replays log',
            'Atomicity: Complex operations logged together, replay ensures all-or-nothing',
            'Sequential writes: 10-1000x faster than random writes (no disk head movement)',
            'Example: PostgreSQL 10K writes/sec (WAL) vs 100 writes/sec (direct random)',
        ],
    },
    {
        id: 'q2',
        question:
            'Describe the three phases of crash recovery using WAL (Analysis, Redo, Undo) and explain why all three are necessary.',
        sampleAnswer: `WAL-based crash recovery has three phases to restore a consistent state: Analysis Phase: Scan WAL to understand what happened. Identify: (1) Which transactions were committed (have COMMIT record in log). (2) Which transactions were active but not committed. (3) Which pages were dirty (modified but not flushed to disk). Output: List of committed transactions to redo, uncommitted transactions to undo, dirty pages. Redo Phase: Replay all committed transactions from WAL. For each logged operation (from committed transactions), reapply the change to data structures. This ensures durability—even if page wasn't flushed before crash, redo makes the change persistent.Important: Redo is idempotent(can apply same operation multiple times safely).Undo Phase: Roll back uncommitted transactions.Use "before" images from WAL to reverse changes.Ensures atomicity—incomplete transactions are completely removed.Example: Transaction T1 committed(redo: apply changes).Transaction T2 started but not committed(undo: reverse changes).Why all three?(1) Analysis determines what to do(which transactions need redo/ undo). (2) Redo ensures durability(committed work survives). (3) Undo ensures atomicity(uncommitted work removed).Without any phase, system could be in inconsistent state—partially applied transactions or lost committed work.`,
        keyPoints: [
            'Analysis: Scan WAL to identify committed transactions, active transactions, dirty pages',
            'Redo: Replay committed transactions, ensures durability of acknowledged operations',
            'Undo: Roll back uncommitted transactions using "before" images, ensures atomicity',
            'Why all three: Analysis determines plan, Redo ensures durability, Undo ensures atomicity',
            'Result: Consistent state with all committed work and no partial transactions',
        ],
    },
    {
        id: 'q3',
        question:
            'Explain checkpointing in WAL systems. What problem does it solve, and what are the trade-offs between consistent and fuzzy checkpoints?',
        sampleAnswer: `Problem: WAL grows indefinitely without bounds. During recovery, must replay entire log from beginning, which takes arbitrarily long as log grows. Checkpointing solves this by creating periodic snapshots of database state, allowing truncation of old log entries and faster recovery. How checkpointing works: (1) Flush all dirty pages (in-memory changes) to disk. (2) Write checkpoint record to WAL, noting LSN and active transactions. (3) Truncate log before checkpoint (those entries now redundant—reflected in on-disk state). (4) During recovery, start from last checkpoint, not beginning. Consistent Checkpoint: Stop all writes, flush all dirty pages, write checkpoint, resume. Pros: Simple implementation, checkpoint represents exact point-in-time state. Cons: Downtime during checkpoint (seconds to minutes for large databases)—unacceptable for online systems. Fuzzy Checkpoint: Allow writes to continue during checkpointing. Record LSN at checkpoint start. Flush dirty pages asynchronously (may take time). Checkpoint record notes: "restore from me, then replay from this LSN." Pros: No downtime, suitable for production systems. Cons: Recovery slightly more complex (must replay some operations after checkpoint). Trade-offs: Consistent checkpoints are simpler but cause downtime. Fuzzy checkpoints enable continuous availability but add complexity. Production databases (PostgreSQL, MySQL) use fuzzy checkpoints. Checkpoint frequency: Too often = overhead, too rare = long recovery and large logs. Typical: every few GB of WAL or every hour.`,
        keyPoints: [
            'Problem: Unbounded WAL growth, slow recovery (replay entire log)',
            'Solution: Periodic checkpoints flush pages, truncate log, enable fast recovery from checkpoint',
            'Consistent: Stop writes, flush all, checkpoint—simple but causes downtime',
            'Fuzzy: Continue writes during checkpoint—no downtime, slightly complex recovery',
            'Production use fuzzy: Trade simplicity for availability (PostgreSQL, MySQL)',
        ],
    },
];
