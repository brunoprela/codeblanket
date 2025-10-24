/**
 * Multiple choice questions for Read Repair section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const readrepairMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      "In Cassandra's probabilistic read repair with read_repair_chance=0.1, what percentage of reads check ALL replicas for inconsistencies?",
    options: ['1%', '10%', '50%', '100%'],
    correctAnswer: 1,
    explanation:
      'With read_repair_chance=0.1, exactly 10% (0.1 = 10%) of reads check ALL replicas for inconsistencies, while the remaining 90% only read from the quorum. This probabilistic approach balances consistency and performance. The process: 90% of reads (fast path): Read from quorum only (e.g., R=2 out of RF=3). Return result immediately to client. Background: Asynchronously check remaining replicas, repair if needed. 10% of reads (full repair): Read from ALL replicas (all 3, not just quorum 2). Wait for all responses, compare. Repair any stale replicas immediately. Return result to client (includes repair latency). Trade-off: 10% sufficient to catch most inconsistencies over time (frequently accessed data quickly repaired). 90% fast path preserves read performance. Higher values (e.g., 50%) increase consistency at cost of performance (50% of reads slower). Lower values (e.g., 1%) improve performance but slower inconsistency detection. Why 10% default: Balance—catches inconsistencies quickly for hot data without significantly impacting latency. Tunable per table based on requirements (financial data might use 30%, logs might use 1%). Option 1 (1%) too low. Options 3 and 4 too high, would significantly degrade read performance.',
  },
  {
    id: 'mc2',
    question:
      'What is the key difference between synchronous (blocking) and asynchronous read repair?',
    options: [
      'Synchronous uses TCP; asynchronous uses UDP',
      'Synchronous waits for repair to complete before returning to client; asynchronous repairs in background',
      'Synchronous repairs all replicas; asynchronous repairs only one',
      'Synchronous is faster than asynchronous',
    ],
    correctAnswer: 1,
    explanation:
      "The key difference is whether the client waits for repair completion. Synchronous (blocking) read repair: Read from quorum, detect inconsistency. Repair stale replicas. Wait for repair acknowledgments. Return result to client. Client latency includes repair time (e.g., 50ms vs normal 10ms). Guarantees client sees result after repair complete. Asynchronous (background) read repair: Read from quorum, detect inconsistency. Return result to client immediately (fast!). In background: Repair stale replicas without blocking. Client latency is normal (~10ms). Client might get result before repair completes. Example: Cassandra blocking read repair (always on): Compares quorum responses. If mismatch, repairs and waits. Guarantees consistent response. Cassandra probabilistic read repair (10% of reads): Reads all replicas, repairs, waits (blocks). Other 90% use async repair (background). Trade-offs: Synchronous: Stronger consistency, higher latency. Asynchronous: Lower latency, eventually consistent (repair happens but client doesn't wait). Production typically uses synchronous for quorum and async for remainder to balance both. Option 1 (protocol) is incorrect. Option 3 (number of replicas) is incorrect. Option 4 (speed) is backward—async is faster for client.",
  },
  {
    id: 'mc3',
    question:
      'Why is read repair alone insufficient to guarantee consistency for all data in a distributed database?',
    options: [
      'Read repair is too slow',
      'Read repair only fixes data that is read, leaving cold (rarely read) data potentially inconsistent',
      'Read repair causes data corruption',
      'Read repair only works on small datasets',
    ],
    correctAnswer: 1,
    explanation:
      "Read repair only fixes keys that are actually read, creating a blind spot for cold data (infrequently accessed keys) that may remain inconsistent indefinitely. The problem: Dataset: 1 billion keys. Read distribution: 1% of keys account for 90% of reads (hot data, power law). Inconsistency: 100K keys inconsistent across replicas. With only read repair: Hot keys (~900K inconsistent keys that are read): Fixed quickly (hours/days as users access them). Cold keys (~99K inconsistent keys rarely/never read): May never be read. Remain inconsistent for months/years. User eventually accesses cold key (rare): Gets stale data (bad user experience). Example: Old user account (never logs in). Account data inconsistent across replicas. Years later, user logs in. Read repair fixes it then, but user may have seen wrong data. This is why production systems use both: Read repair: Fixes hot data automatically (fast, automatic). Anti-entropy: Periodically scans ALL data, fixes cold data (slow, comprehensive). Together: Most data fixed quickly (read repair), completeness guaranteed (anti-entropy). Option 1 (slow) is incorrect—read repair is fast for hot data. Option 3 (corruption) is opposite—read repair prevents corruption. Option 4 (size) is incorrect—scales fine, just doesn't cover all keys.",
  },
  {
    id: 'mc4',
    question:
      'In a read repair scenario where concurrent writes occur during the repair, how is the correct data determined?',
    options: [
      'The first write always wins',
      'The repair is cancelled and retried',
      'Timestamps or version vectors are compared at application time, with newer data taking precedence',
      'A random value is chosen',
    ],
    correctAnswer: 2,
    explanation:
      'Concurrent writes during repair are resolved by comparing timestamps or version vectors when applying the repair, ensuring the newest data is preserved. The race condition: T=0: Read detects replica C has stale X=1. T=1: Coordinator decides to repair C with X=2. T=2: New write X=3 arrives, goes to all replicas. T=3: C now has X=3 (newest). T=4: Repair message arrives at C: "update to X=2." T=5: C must compare before applying. Version vector comparison: Repair vector: {A:2, B:2, C:1} (from before concurrent write). Local vector: {A:3, B:3, C:2} (includes concurrent write). Local dominates repair: Reject repair, keep X=3. Timestamp comparison: Repair timestamp: 100. Local timestamp: 150 (concurrent write). Local newer: Reject repair, keep X=3. If timestamps/vectors were ignored: Apply repair X=2 blindly. Overwrite X=3 (newest) with X=2 (stale). Data regression! By checking at application time, the system ensures correct ordering even with concurrent operations. This makes repair idempotent and safe—can replay repairs without risk of data loss. Options 1 and 4 (arbitrary choice) would cause inconsistency. Option 2 (cancel retry) is wasteful—no need to retry if local data is newer.',
  },
  {
    id: 'mc5',
    question:
      'What is the trade-off of using a high read_repair_chance (e.g., 50%) versus a low value (e.g., 1%)?',
    options: [
      'High value increases consistency but decreases read performance; low value does the opposite',
      'High value uses more memory; low value uses more CPU',
      'High value works only on small clusters; low value works on large clusters',
      'High value requires manual intervention; low value is automatic',
    ],
    correctAnswer: 0,
    explanation:
      'The trade-off is consistency versus read performance: higher read_repair_chance means more reads check all replicas (better consistency but slower), while lower values mean fewer checks (faster reads but slower inconsistency detection). High read_repair_chance (50%): Pros: Faster inconsistency detection (50% of reads check all replicas). Higher data consistency across cluster. Less reliance on anti-entropy. Cons: 50% of reads experience higher latency (wait for all replicas, not just quorum). Increased network traffic (read from all replicas more often). Reduced read throughput (slower individual reads). Example: Read normally 10ms (quorum), but 50% take 50ms (all replicas + repair). Average latency: 0.5×10 + 0.5×50 = 30ms (3× slower). Low read_repair_chance (1%): Pros: 99% of reads are fast (quorum only). Lower network overhead. Higher read throughput. Cons: Slower inconsistency detection (only 1% check all replicas). Lower consistency (rely more on anti-entropy). More stale reads possible. Example: Average latency: 0.99×10 + 0.01×50 = 10.4ms (minimal impact). When to use high: Critical data (financial), consistency paramount, acceptable performance hit. When to use low: High-volume reads (analytics), eventual consistency acceptable, performance critical. Production tuning: Financial tables: 20-30%. Log/analytics tables: 1-5%. Frequently updated tables: Higher (more inconsistencies expected). Rarely updated: Lower (inconsistencies rare). Options 2, 3, and 4 are incorrect—not related to the actual trade-off.',
  },
];
