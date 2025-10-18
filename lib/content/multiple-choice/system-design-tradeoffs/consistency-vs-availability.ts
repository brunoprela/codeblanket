/**
 * Multiple choice questions for Consistency vs Availability section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const consistencyvsavailabilityMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'According to CAP theorem, during a network partition, you must choose between:',
      options: [
        'Consistency and Partition Tolerance',
        'Availability and Partition Tolerance',
        'Consistency and Availability',
        'All three can be maintained',
      ],
      correctAnswer: 2,
      explanation:
        'During a network partition (P is present), you must choose between Consistency (all nodes see same data) and Availability (always respond to requests). You cannot have both during a partition. Partition Tolerance is required in distributed systems, so the real choice is C or A.',
    },
    {
      id: 'mc2',
      question:
        'Which type of system should prioritize consistency over availability?',
      options: [
        'Social media news feed',
        'Banking transaction processing',
        'Product recommendation engine',
        'Analytics dashboard',
      ],
      correctAnswer: 1,
      explanation:
        'Banking transactions require strong consistency. You cannot show incorrect account balances or process payments twice. It is better to deny service (unavailable) than to show inconsistent financial data. Social feeds, recommendations, and analytics can tolerate eventual consistency.',
    },
    {
      id: 'mc3',
      question:
        'In Cassandra with RF=3, to achieve strong consistency, what should W + R equal?',
      options: [
        'W + R = 2',
        'W + R = 3',
        'W + R > 3 (e.g., W=2, R=2)',
        'W + R = 1',
      ],
      correctAnswer: 2,
      explanation:
        'For strong consistency in Cassandra, W + R > N (where N is replication factor). With RF=3, using W=2 and R=2 gives W+R=4 > 3, ensuring that read and write quorums overlap, guaranteeing you read the most recent write. This is called quorum consistency.',
    },
    {
      id: 'mc4',
      question:
        'Which database is designed as an AP (Available, Partition-tolerant) system?',
      options: ['PostgreSQL', 'MySQL', 'Cassandra', 'Redis (single instance)'],
      correctAnswer: 2,
      explanation:
        'Cassandra is designed as an AP system that prioritizes availability and partition tolerance. It uses eventual consistency by default and continues to serve requests even during network partitions. PostgreSQL, MySQL, and single-instance Redis prioritize consistency (CP).',
    },
    {
      id: 'mc5',
      question:
        'What is the main trade-off when choosing strong consistency (CP) over availability (AP)?',
      options: [
        'Lower development cost',
        'Simpler architecture',
        'Higher latency and potential service unavailability',
        'Better user experience',
      ],
      correctAnswer: 2,
      explanation:
        'Strong consistency (CP) requires coordination between nodes, which increases latency. During network partitions, the system may become unavailable to maintain consistency. This trades user experience (speed, availability) for data correctness. AP systems have lower latency and higher availability but may serve stale data.',
    },
  ];
