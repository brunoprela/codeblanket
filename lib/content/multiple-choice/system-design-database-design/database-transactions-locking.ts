/**
 * Multiple choice questions for Database Transactions & Locking section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const databasetransactionslockingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'trans-1',
      question:
        'Which isolation level prevents dirty reads but allows non-repeatable reads and phantom reads?',
      options: [
        'Read Uncommitted',
        'Read Committed',
        'Repeatable Read',
        'Serializable',
      ],
      correctAnswer: 1,
      explanation:
        'Option B (Read Committed) is correct. This isolation level ensures you only read committed data (no dirty reads), but if you read the same row twice within a transaction, you might see different values if another transaction commits a change in between (non-repeatable read). Phantom reads (new rows appearing in range queries) are also possible. This is the default isolation level for most databases like PostgreSQL, SQL Server, and Oracle. Read Uncommitted allows dirty reads. Repeatable Read prevents non-repeatable reads. Serializable prevents all anomalies.',
    },
    {
      id: 'trans-2',
      question:
        'What is the main difference between FOR UPDATE and FOR SHARE in SELECT statements?',
      options: [
        'FOR UPDATE acquires a shared lock, FOR SHARE acquires an exclusive lock',
        'FOR UPDATE acquires an exclusive lock, FOR SHARE acquires a shared lock',
        'FOR UPDATE locks the entire table, FOR SHARE locks individual rows',
        'FOR UPDATE prevents reads, FOR SHARE prevents only writes',
      ],
      correctAnswer: 1,
      explanation:
        'Option B is correct. FOR UPDATE acquires an exclusive lock (X lock), which blocks both reads and writes from other transactions. FOR SHARE acquires a shared lock (S lock), which allows other transactions to also acquire shared locks (multiple readers) but blocks exclusive locks (writers). Option A is backwards. Option C is incorrect (both are row-level by default). Option D is close but imprecise - FOR UPDATE blocks other transactions from acquiring conflicting locks, not literally preventing all reads (depends on isolation level and lock compatibility).',
    },
    {
      id: 'trans-3',
      question:
        'You have an e-commerce site where thousands of users might try to buy the last item in stock. Which approach is most appropriate?',
      options: [
        'Read Uncommitted isolation level for maximum performance',
        'Pessimistic locking with FOR UPDATE to guarantee no overselling',
        'No locks; check stock after order creation',
        'Optimistic locking with version column for better concurrency',
      ],
      correctAnswer: 1,
      explanation:
        'Option B (pessimistic locking) is most appropriate for this high-contention scenario. When many users compete for limited stock, pessimistic locking ensures that once a transaction acquires the lock on a product row, no other transaction can proceed until it commits or rolls back. This guarantees no overselling. Option A (Read Uncommitted) would allow dirty reads and race conditions. Option C creates race conditions where multiple users could pass the stock check. Option D (optimistic locking) would cause many failed transactions and retries under high contention, degrading user experience. For high-contention resources, pessimistic locking is preferred despite lower concurrency.',
    },
    {
      id: 'trans-4',
      question:
        'What is the best strategy to prevent deadlocks in a system where transactions frequently update multiple accounts?',
      options: [
        'Use the highest isolation level (Serializable) to prevent conflicts',
        'Acquire locks in a consistent order (e.g., always lock lower account_id first)',
        'Use longer transactions to reduce the number of commits',
        'Disable automatic deadlock detection to improve performance',
      ],
      correctAnswer: 1,
      explanation:
        'Option B is correct. Acquiring locks in a consistent order prevents circular wait conditions that cause deadlocks. For example, if all transactions always lock accounts in ascending order of account_id, no cycle can form. Option A (Serializable) can actually increase deadlock probability due to stricter locking. Option C (longer transactions) increases deadlock probability by holding locks longer. Option D (disabling deadlock detection) is not possible and would cause transactions to hang indefinitely rather than failing fast. The key to deadlock prevention is eliminating cycles in the wait-for graph, achieved through consistent lock ordering.',
    },
    {
      id: 'trans-5',
      question:
        'In a job queue processed by multiple workers, what query pattern prevents workers from competing for the same job?',
      options: [
        'SELECT * FROM jobs WHERE status = "pending" LIMIT 1',
        'SELECT * FROM jobs WHERE status = "pending" LIMIT 1 FOR UPDATE',
        'SELECT * FROM jobs WHERE status = "pending" LIMIT 1 FOR UPDATE SKIP LOCKED',
        'SELECT * FROM jobs WHERE status = "pending" LIMIT 1 FOR SHARE',
      ],
      correctAnswer: 2,
      explanation:
        'Option C (FOR UPDATE SKIP LOCKED) is correct. This pattern allows each worker to immediately acquire the next available unlocked job without waiting. If a job is locked (being processed by another worker), SKIP LOCKED tells the database to skip it and return the next unlocked job. Option A has no locking (race condition). Option B uses FOR UPDATE but without SKIP LOCKED, workers would queue up waiting for locked jobs instead of moving to the next available one. Option D (FOR SHARE) allows multiple workers to read the same job, creating race conditions. SKIP LOCKED (PostgreSQL 9.5+, MySQL 8+) is essential for efficient job queue implementations.',
    },
  ];
