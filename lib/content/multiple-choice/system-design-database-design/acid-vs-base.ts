/**
 * Multiple choice questions for ACID vs BASE Properties section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const acidvsbaseMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'acid-base-q1',
    question:
      "In an e-commerce system, a user completes checkout. The system must: (1) deduct inventory, (2) create order record, (3) charge payment. Step 3 fails. What ACID property ensures the system doesn't leave inventory deducted without a completed order?",
    options: [
      'Consistency - ensures business rules are followed',
      'Isolation - prevents other transactions from interfering',
      'Atomicity - ensures all steps succeed or all fail',
      'Durability - ensures the order is saved permanently',
    ],
    correctAnswer: 2,
    explanation:
      "Atomicity is the correct answer. Atomicity ensures that a transaction is treated as a single unit - either all operations succeed (commit) or all fail (rollback). In this case, if payment (step 3) fails, atomicity ensures that the inventory deduction (step 1) and order creation (step 2) are rolled back. This prevents the system from being in an inconsistent state where inventory is deducted but no order exists and no payment was made. Consistency ensures business rules are followed but doesn't specifically handle partial transaction failures. Isolation prevents concurrent transactions from interfering. Durability ensures committed transactions survive crashes.",
    difficulty: 'medium',
  },
  {
    id: 'acid-base-q2',
    question:
      'Which isolation level prevents "dirty reads" but allows "non-repeatable reads"?',
    options: [
      'Read Uncommitted',
      'Read Committed',
      'Repeatable Read',
      'Serializable',
    ],
    correctAnswer: 1,
    explanation:
      'Read Committed is the correct answer. It prevents dirty reads (reading uncommitted changes from other transactions) but allows non-repeatable reads (same query returning different results within a transaction if another transaction commits changes). Read Uncommitted allows dirty reads. Repeatable Read prevents both dirty reads and non-repeatable reads but allows phantom reads. Serializable prevents all anomalies. Read Committed is the default isolation level in PostgreSQL and Oracle, providing a good balance between consistency and performance.',
    difficulty: 'hard',
  },
  {
    id: 'acid-base-q3',
    question:
      "Netflix uses Cassandra (BASE) for its content catalog. During a network partition, users in Europe can still browse movies even though the latest catalog updates from the US haven't replicated yet. Which BASE property is being demonstrated?",
    options: [
      'Atomicity - operations are atomic',
      'Consistency - all nodes have same data',
      'Basically Available - system responds even with stale data',
      'Durability - data persists after failures',
    ],
    correctAnswer: 2,
    explanation:
      "Basically Available is the correct BASE property being demonstrated. \"Basically Available\" means the system guarantees availability by responding to queries even if it returns stale or incomplete data. In this scenario, European users can still browse the movie catalog during a partition, even though they're seeing slightly outdated information. This is better than the alternative (system being unavailable) for Netflix's use case. Option A and D are ACID properties, not BASE. Option B (Consistency) is actually what's being sacrificed - the nodes don't have the same data, but availability is maintained.",
    difficulty: 'medium',
  },
  {
    id: 'acid-base-q4',
    question:
      'Your banking application uses PostgreSQL with Serializable isolation level for all transactions. Users complain the system is slow. What trade-off is being made?',
    options: [
      'Sacrificing durability for performance',
      'Sacrificing consistency for availability',
      'Sacrificing performance for strongest isolation guarantees',
      'Sacrificing atomicity for scalability',
    ],
    correctAnswer: 2,
    explanation:
      'Sacrificing performance for strongest isolation is correct. Serializable isolation provides the strongest guarantees by ensuring transactions execute as if they were serial (one after another). This prevents all concurrency anomalies (dirty reads, non-repeatable reads, phantom reads) but comes at a significant performance cost due to increased locking and potential for conflicts/retries. For a banking application, this trade-off is often appropriate because correctness is more important than speed. However, if the performance impact is too severe, you might consider downgrading to Repeatable Read or Read Committed for less critical operations. Durability, consistency, and atomicity are not being sacrificed.',
    difficulty: 'hard',
  },
  {
    id: 'acid-base-q5',
    question:
      'Which of the following systems REQUIRES ACID properties and would be problematic with BASE (eventual consistency)?',
    options: [
      'Social media like counter showing "423 likes"',
      'Product catalog showing product descriptions',
      'Flight booking system where each seat can only be sold once',
      'News article comment section',
    ],
    correctAnswer: 2,
    explanation:
      'Flight booking system requires ACID properties. Selling each seat only once requires atomicity (check availability + book seat must be atomic) and strong consistency (all booking systems must see seat as unavailable immediately after booking). With BASE/eventual consistency, you could have two users book the same seat during replication lag, leading to double-booking. Social media like counters can tolerate being off by a few (eventual consistency fine). Product catalogs rarely change and can show slightly stale data. Comment sections can handle eventual consistency (comments appearing a few seconds late is acceptable).',
    difficulty: 'medium',
  },
];
