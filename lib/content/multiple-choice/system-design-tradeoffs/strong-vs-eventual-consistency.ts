/**
 * Multiple choice questions for Strong Consistency vs Eventual Consistency section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const strongvseventualconsistencyMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What does strong consistency guarantee?',
      options: [
        'All writes eventually succeed',
        'All reads return the most recent write',
        'The system is always available',
        'Conflicts never occur',
      ],
      correctAnswer: 1,
      explanation:
        'Strong consistency (linearizability) guarantees that all reads return the most recent write. After a write completes, all subsequent reads see that write or a newer one. This makes the system behave as if there is only one copy of the data.',
    },
    {
      id: 'mc2',
      question:
        'Which consistency model is appropriate for a like count on a social media post?',
      options: [
        'Strong consistency (must be exact at all times)',
        'Eventual consistency (approximate count acceptable)',
        'No consistency needed',
        'Synchronous replication required',
      ],
      correctAnswer: 1,
      explanation:
        "Like counts are non-critical and can use eventual consistency. Users don't care if the count shows 1,234 vs 1,237 briefly. Eventually all regions will converge to the correct count. This enables high availability and low latency, which matters more than perfect accuracy for likes.",
    },
    {
      id: 'mc3',
      question: 'What is "read-your-writes" consistency?',
      options: [
        'All users see writes immediately',
        'A user always sees their own writes in subsequent reads',
        'Writes are always successful',
        'Reads are faster than writes',
      ],
      correctAnswer: 1,
      explanation:
        'Read-your-writes consistency guarantees that after a user writes data, their subsequent reads will see that write (or newer). This is critical for UX in eventually consistent systems - users expect to see their own changes immediately even if other users see them slightly later.',
    },
    {
      id: 'mc4',
      question:
        'How does eventual consistency achieve higher availability than strong consistency?',
      options: [
        'It uses faster hardware',
        'It accepts writes even when some replicas are unavailable',
        'It never has network partitions',
        'It caches all data in memory',
      ],
      correctAnswer: 1,
      explanation:
        "Eventual consistency achieves higher availability by accepting writes even when some replicas are unavailable or unreachable. It doesn't require coordination between all nodes before confirming writes. Strong consistency must coordinate with multiple nodes, so if any are down, writes may fail.",
    },
    {
      id: 'mc5',
      question:
        'What is the primary trade-off when choosing eventual consistency over strong consistency?',
      options: [
        'Higher cost',
        'More complex architecture',
        'Temporary stale reads and need for conflict resolution',
        'Lower security',
      ],
      correctAnswer: 2,
      explanation:
        'The primary trade-off of eventual consistency is that reads may temporarily return stale data, and you need conflict resolution logic for concurrent writes. In exchange, you get lower latency, higher availability, and better scalability. The application must handle temporary inconsistencies.',
    },
  ];
