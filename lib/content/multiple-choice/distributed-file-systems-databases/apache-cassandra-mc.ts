/**
 * Multiple choice questions for Apache Cassandra section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const cassandraMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the formula for achieving strong consistency in Cassandra?',
    options: [
      'Write CL = Read CL',
      'Write CL + Read CL > Replication Factor',
      'Write CL = ALL',
      'Read CL = ONE',
    ],
    correctAnswer: 1,
    explanation:
      'Strong consistency in Cassandra: Write CL + Read CL > RF. Example: RF=3, QUORUM write (2 nodes) + QUORUM read (2 nodes) = 4 > 3. This guarantees read sees at least one node with the latest write. Common: LOCAL_QUORUM for both operations.',
  },
  {
    id: 'mc2',
    question: 'What is the role of the coordinator node in Cassandra?',
    options: [
      'Store all data',
      'Be the master node',
      'Route requests and coordinate operations (any node can be coordinator)',
      'Only handle authentication',
    ],
    correctAnswer: 2,
    explanation:
      'In Cassandra, ANY node can act as coordinator. The node that receives a client request becomes the coordinator for that operation. Coordinator determines replicas (via consistent hashing), forwards requests, collects responses, and returns result to client. No dedicated coordinator node.',
  },
  {
    id: 'mc3',
    question: 'What is hinted handoff in Cassandra?',
    options: [
      'Data encryption method',
      'Storing writes temporarily when replica is down, replaying when it recovers',
      'Query optimization technique',
      'Backup strategy',
    ],
    correctAnswer: 1,
    explanation:
      'Hinted handoff: When target replica is temporarily unavailable, coordinator stores a "hint" on another node. When target recovers, hints are replayed to catch it up. Improves eventual consistency and reduces repair workload. Hints have TTL (3 hours default). Not a substitute for repair.',
  },
  {
    id: 'mc4',
    question:
      'What compaction strategy should you use for time-series data in Cassandra?',
    options: [
      'SizeTiered Compaction (STCS)',
      'Leveled Compaction (LCS)',
      'Time-Window Compaction (TWCS)',
      'No compaction needed',
    ],
    correctAnswer: 2,
    explanation:
      'Time-Window Compaction (TWCS) is designed for time-series data. It compacts data in time windows (e.g., hourly, daily). Old windows are never recompacted with new data, making it efficient for time-series where old data is rarely updated. Prevents write amplification for time-series workloads.',
  },
  {
    id: 'mc5',
    question: 'How does Cassandra achieve high availability?',
    options: [
      'Single master with hot standby',
      'Masterless peer-to-peer architecture with tunable replication',
      'RAID arrays only',
      'Daily backups',
    ],
    correctAnswer: 1,
    explanation:
      'Cassandra achieves high availability through: (1) Masterless architecture (no single point of failure), (2) Replication across multiple nodes (RF=3 typical), (3) Tunable consistency (can read/write even with node failures), (4) Multi-datacenter support. Any node can serve requests. Cluster continues operating even with multiple node failures.',
  },
];
