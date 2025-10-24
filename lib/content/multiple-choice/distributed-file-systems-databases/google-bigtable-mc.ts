/**
 * Multiple choice questions for Google BigTable section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const bigtableMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the data model of BigTable?',
    options: [
      'Relational tables with foreign keys',
      'Sparse, distributed, persistent multi-dimensional sorted map',
      'Key-value store only',
      'Graph database',
    ],
    correctAnswer: 1,
    explanation:
      "BigTable data model: (row key, column key, timestamp) → value. It's a sparse map (empty cells don't consume space), distributed (across tablet servers), persistent (stored in GFS), multi-dimensional (row, column, time), and sorted (by row key). Not relational!",
  },
  {
    id: 'mc2',
    question: 'What is an SSTable in BigTable?',
    options: [
      'A database query language',
      'An immutable, sorted, compressed file stored in GFS',
      'A type of index',
      'A replication protocol',
    ],
    correctAnswer: 1,
    explanation:
      'SSTable (Sorted String Table) is an immutable file format in BigTable. Once created, never modified. Contains sorted key-value pairs, block-compressed, with index for fast lookups. Stored in GFS for durability and replication. Multiple SSTables accumulate over time, requiring compaction.',
  },
  {
    id: 'mc3',
    question: 'What is the purpose of compaction in BigTable?',
    options: [
      'Encrypt data',
      'Merge multiple SSTables, remove deleted cells and old versions, reclaim space',
      'Replicate data across datacenters',
      'Index data for faster queries',
    ],
    correctAnswer: 1,
    explanation:
      'Compaction merges multiple SSTables into fewer, larger SSTables. Removes deleted entries (tombstones) and old versions, reclaims disk space, and improves read performance by reducing the number of SSTables to check. Minor compaction (frequent) and major compaction (infrequent, expensive).',
  },
  {
    id: 'mc4',
    question: 'What role does Chubby play in BigTable?',
    options: [
      'Store all user data',
      'Distributed lock service providing coordination and storing root tablet location',
      'Query optimizer',
      'Data compression',
    ],
    correctAnswer: 1,
    explanation:
      "Chubby is Google's distributed lock service (Paxos-based) used by BigTable for: (1) Ensuring one active master, (2) Storing root tablet location, (3) Tablet server lease management, (4) Cluster coordination. Chubby provides consistency and coordination, not data storage.",
  },
  {
    id: 'mc5',
    question: 'What is a memtable in BigTable?',
    options: [
      'A disk-based cache',
      'An in-memory sorted buffer for recent writes',
      'A network protocol',
      'A compression algorithm',
    ],
    correctAnswer: 1,
    explanation:
      "Memtable is an in-memory sorted buffer that accumulates recent writes. Writes go to commit log (durability) and memtable (fast access). When memtable reaches size limit (~64 MB), it's flushed to an immutable SSTable on GFS. Memtable enables fast writes and serves recent reads.",
  },
];
