/**
 * Multiple choice questions for Consistent Hashing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const consistenthashingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'When using modulo hashing with 100 servers, you add a 101st server. Approximately what percentage of keys need to be remapped?',
    options: ['1%', '25%', '50%', '99%'],
    correctAnswer: 3,
    explanation:
      'With modulo hashing (hash % N), changing N from 100 to 101 means almost every key will hash to a different server. Formula: n/(n+1) = 100/101 ≈ 99% of keys remap. This is why modulo hashing does not scale for distributed systems. Consistent hashing remaps only ~1% of keys in this scenario.',
  },
  {
    id: 'mc2',
    question: 'In consistent hashing, a key is stored on which server?',
    options: [
      'The server with the closest hash value',
      'The first server encountered when moving clockwise on the ring',
      'The server with the smallest hash value',
      'A random server',
    ],
    correctAnswer: 1,
    explanation:
      'In consistent hashing, after hashing the key to a position on the ring, we move clockwise from that position and store the key on the FIRST server we encounter. This ensures deterministic routing while minimizing remapping when servers are added or removed.',
  },
  {
    id: 'mc3',
    question: 'What is the purpose of virtual nodes in consistent hashing?',
    options: [
      'To increase storage capacity',
      'To improve security',
      'To achieve more even load distribution across servers',
      'To reduce network latency',
    ],
    correctAnswer: 2,
    explanation:
      'Virtual nodes solve the load balancing problem. Without them, random hash placement can cause one server to handle 40% of load while others handle 30%. By placing each physical server 100-200 times on the ring (as virtual nodes), we achieve statistical averaging that ensures ~equal load distribution across all servers (<5% variance).',
  },
  {
    id: 'mc4',
    question:
      'How many virtual nodes per physical server is typically recommended?',
    options: [
      '5-10 virtual nodes',
      '20-30 virtual nodes',
      '100-200 virtual nodes',
      '1000+ virtual nodes',
    ],
    correctAnswer: 2,
    explanation:
      'Industry standard is 100-200 virtual nodes per physical server. Cassandra default: 256. DynamoDB: 100-200. This provides good load distribution (<5% variance) without excessive memory/CPU overhead. More vnodes = better distribution but diminishing returns beyond 150-200.',
  },
  {
    id: 'mc5',
    question: 'Which distributed systems use consistent hashing in production?',
    options: [
      'Only academic research systems',
      'Amazon DynamoDB, Apache Cassandra, Redis Cluster',
      'Only MySQL and PostgreSQL',
      'Only in-memory caches',
    ],
    correctAnswer: 1,
    explanation:
      'Consistent hashing is the proven industry standard: Amazon DynamoDB (partitioning), Apache Cassandra (token ring), Redis Cluster (key distribution), Memcached (with libmemcached), Discord (guild routing), Akamai CDN (content distribution). It is the fundamental technique for scaling distributed systems, not academic theory.',
  },
];
