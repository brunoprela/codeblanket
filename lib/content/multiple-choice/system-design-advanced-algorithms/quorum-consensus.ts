/**
 * Multiple choice questions for Quorum Consensus section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const quorumconsensusMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'For N=5 replicas, which configuration guarantees strong consistency?',
    options: ['W=1, R=1', 'W=2, R=2', 'W=3, R=3', 'All of the above'],
    correctAnswer: 2,
    explanation:
      'Strong consistency requires W + R > N. With N=5: (1) W=1,R=1: 1+1=2≤5 ✗. (2) W=2,R=2: 2+2=4≤5 ✗. (3) W=3,R=3: 3+3=6>5 ✓. Only option 3 satisfies the quorum rule. This ensures read and write quorums must overlap, guaranteeing reads see latest writes.',
  },
  {
    id: 'mc2',
    question:
      'You have N=3 replicas with W=2, R=2. Two servers go down simultaneously. What happens to reads and writes?',
    options: [
      'Both reads and writes succeed',
      'Reads succeed, writes fail',
      'Writes succeed, reads fail',
      'Both reads and writes fail',
    ],
    correctAnswer: 3,
    explanation:
      'With only 1 server available: Writes need W=2 servers but only 1 available → writes FAIL. Reads need R=2 servers but only 1 available → reads FAIL. This demonstrates the availability trade-off: quorum ensures consistency but requires majority of replicas to be available. With strict quorum, losing 2 of 3 servers means no operations succeed.',
  },
  {
    id: 'mc3',
    question: 'What is the advantage of sloppy quorum over strict quorum?',
    options: [
      'Stronger consistency guarantees',
      'Lower latency',
      'Higher availability during failures',
      'Simpler implementation',
    ],
    correctAnswer: 2,
    explanation:
      'Sloppy quorum improves availability by allowing writes to ANY N healthy servers (not just designated replicas). If designated replicas are down, writes succeed on temporary replicas with hinted handoff. Trade-off: eventual consistency instead of strong consistency. Used by DynamoDB and Riak for high availability.',
  },
  {
    id: 'mc4',
    question:
      'For a read-heavy workload with N=5 replicas, which configuration is optimal while maintaining strong consistency?',
    options: ['W=1, R=1', 'W=2, R=4', 'W=3, R=3', 'W=4, R=2'],
    correctAnswer: 3,
    explanation:
      'For read-heavy workloads, optimize for low R (fast reads). W=4,R=2 means: (1) Reads only need 2 responses (fast). (2) Writes need 4 responses (slower, but acceptable for rare writes). (3) W+R=6>5 maintains strong consistency. This configuration optimizes for the common case (reads) while maintaining correctness.',
  },
  {
    id: 'mc5',
    question:
      'Which distributed systems use quorum-based replication in production?',
    options: [
      'Only academic research systems',
      'Apache Cassandra and Amazon DynamoDB',
      'Only traditional SQL databases',
      'Only in-memory caches',
    ],
    correctAnswer: 1,
    explanation:
      'Quorum-based replication is the industry standard: Apache Cassandra (configurable consistency levels), Amazon DynamoDB (W=2,R=2 default), Riak, MongoDB (write/read concerns), Voldemort. Nearly every distributed database uses quorum consensus to balance consistency and availability. This is proven production technology, not theory.',
  },
];
