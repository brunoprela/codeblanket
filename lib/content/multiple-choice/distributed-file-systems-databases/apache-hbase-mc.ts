/**
 * Multiple choice questions for Apache HBase section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hbaseMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What consistency model does HBase provide?',
    options: [
      'Eventual consistency',
      'Causal consistency',
      'Strong consistency',
      'No consistency guarantees',
    ],
    correctAnswer: 2,
    explanation:
      'HBase provides strong consistency because each row is served by a single RegionServer. Writes go to WAL and memstore, immediately readable. No distributed coordination needed for single-row operations. This is unlike Cassandra which provides tunable eventual consistency. Trade-off: Strong consistency vs slightly lower availability.',
  },
  {
    id: 'mc2',
    question: 'What is the purpose of the Write-Ahead Log (WAL) in HBase?',
    options: [
      'Query optimization',
      'Data compression',
      'Durability - ensures writes survive crashes before memstore flush',
      'Load balancing',
    ],
    correctAnswer: 2,
    explanation:
      'WAL ensures durability. Writes go to WAL (on HDFS) first, then memstore. If RegionServer crashes before memstore flush, WAL is replayed to recover data. WAL is sequential append (fast). Without WAL, in-memory data would be lost on crash. Essential for data durability.',
  },
  {
    id: 'mc3',
    question: 'What is a Region in HBase?',
    options: [
      'A geographic location',
      'A contiguous range of rows served by a RegionServer',
      'A backup copy',
      'A query type',
    ],
    correctAnswer: 1,
    explanation:
      'Region is a contiguous range of rows (like BigTable tablets). Initial table = 1 region, splits when reaches size threshold (default 10 GB). Each region served by one RegionServer. Regions are the unit of distribution, load balancing, and recovery. Master assigns regions to RegionServers.',
  },
  {
    id: 'mc4',
    question: 'What role does ZooKeeper play in HBase?',
    options: [
      'Store all data',
      'Coordinate cluster, track HMaster, provide region metadata',
      'Execute queries',
      'Compress data',
    ],
    correctAnswer: 1,
    explanation:
      'ZooKeeper provides coordination for HBase: (1) Track active HMaster (election), (2) Store region assignment state, (3) Detect RegionServer failures (heartbeat), (4) Store cluster configuration. ZooKeeper is critical for HBase coordination and fault tolerance. Without ZK, HBase cannot function.',
  },
  {
    id: 'mc5',
    question:
      'What is the difference between minor and major compaction in HBase?',
    options: [
      'No difference',
      'Minor: merge few HFiles. Major: merge all HFiles, remove tombstones',
      'Minor: fast, Major: slow (no other differences)',
      'Minor: automatic, Major: manual only',
    ],
    correctAnswer: 1,
    explanation:
      "Minor compaction: Merge smaller HFiles into larger ones, runs frequently, doesn't remove deleted cells (tombstones). Major compaction: Merge ALL HFiles in region, remove tombstones and old versions, reclaim space, improve read performance. Expensive, runs weekly/monthly. Major is critical but resource-intensive.",
  },
];
