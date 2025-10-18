/**
 * Multiple choice questions for Database Replication section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const databasereplicationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Your application has 1 primary database and 5 read replicas using asynchronous replication. A user writes data to the primary, then immediately reads from a replica. What might happen?',
    options: [
      'The user will always see their write immediately (strong consistency)',
      'The user might not see their write yet due to replication lag (eventual consistency)',
      'The write will fail because replicas are read-only',
      'The read will automatically be routed to the primary',
    ],
    correctAnswer: 1,
    explanation:
      "With asynchronous replication, the user might not see their write immediately (replication lag 1-5 seconds typical). The write goes to primary and confirms immediately. Replication to replicas happens in background. If user reads from replica before replication completes, they see old data. This is eventual consistency. Solution: Read-your-own-writes pattern (route user's reads to primary for 5-10 seconds after they write).",
  },
  {
    id: 'mc2',
    question:
      'What is the main advantage of synchronous replication over asynchronous replication?',
    options: [
      'Synchronous replication is faster',
      'Synchronous replication has no data loss risk (replica always has latest data)',
      'Synchronous replication works better across geographic regions',
      'Synchronous replication is simpler to implement',
    ],
    correctAnswer: 1,
    explanation:
      'Synchronous replication guarantees no data loss: write confirmed only after data written to both primary and replica. If primary fails immediately after write, data exists on replica. Trade-off: Slower writes (must wait for replica acknowledgment). Asynchronous is faster but risks losing recent writes if primary fails. Synchronous does NOT work well across regions (high latency). Used for critical data (banking, financial).',
  },
  {
    id: 'mc3',
    question:
      'You have 1 primary database handling 5K writes/sec. You add 10 read replicas. What is the write capacity now?',
    options: [
      "5K writes/sec (same as before, replicas don't help writes)",
      '50K writes/sec (10 replicas × 5K each)',
      '55K writes/sec (primary + 10 replicas)',
      '15K writes/sec (primary + average of replicas)',
    ],
    correctAnswer: 0,
    explanation:
      'Read replicas do NOT improve write capacity! All writes still go to single primary (5K writes/sec). Replicas handle READS only (read-only copies of data). To scale writes, need to shard database (split data across multiple primary databases). Replication helps: Read scaling (add replicas → more read capacity). Availability (if primary fails, promote replica). But NOT write scaling.',
  },
  {
    id: 'mc4',
    question:
      'What is split-brain in database replication, and why is it dangerous?',
    options: [
      'The database uses only half its memory (performance issue)',
      'Two databases think they are both primary and accept conflicting writes (data divergence)',
      'Replication lag causes reads to return half the data',
      'The primary database fails and no replica is promoted',
    ],
    correctAnswer: 1,
    explanation:
      'Split-brain: After network partition, both old primary and new primary accept writes → data divergence. Example: Old primary receives Write A (user balance = $100), new primary receives Write B (user balance = $200). When network heals: Which is correct? Data integrity compromised. Prevention: Fencing (STONITH, quorum, epoch numbers) ensures only ONE primary can accept writes. This is critical for data consistency.',
  },
  {
    id: 'mc5',
    question:
      'Your application uses asynchronous replication. The primary database crashes. What data might be lost?',
    options: [
      'No data lost (asynchronous is just as safe as synchronous)',
      'All data in the database is lost',
      "Only the most recent writes (last 1-5 seconds) that weren't replicated yet",
      'Half of the data is lost',
    ],
    correctAnswer: 2,
    explanation:
      'With async replication, only recent writes not yet replicated are lost (typically last 1-5 seconds). Example: Primary receives 1000 writes, replicates 990, crashes → 10 writes lost. Most data is safe (already replicated). This is acceptable for most web apps (social media, e-commerce). For critical data (banking), use synchronous or semi-sync replication to prevent data loss.',
  },
];
