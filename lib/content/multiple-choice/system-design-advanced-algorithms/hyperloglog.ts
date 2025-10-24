/**
 * Multiple choice questions for HyperLogLog section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hyperloglogMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How much memory does HyperLogLog use to count 1 billion unique elements with 0.81% standard error?',
    options: [
      '16 GB (store all elements)',
      '1.6 GB (10% sample)',
      '16 KB (constant size)',
      '16 MB (compressed set)',
    ],
    correctAnswer: 2,
    explanation:
      'HyperLogLog uses constant memory regardless of cardinality. With precision=14, it uses 16,384 buckets × 1 byte = 16 KB total. This gives 0.81% standard error. Exact counting would require ~16 GB to store 1 billion UUIDs. HyperLogLog achieves 1,000,000x memory savings with ~1% error.',
  },
  {
    id: 'mc2',
    question:
      'What is the key operation that allows HyperLogLog estimates to be combined across distributed systems?',
    options: [
      'Add all register values',
      'Take the minimum of each register',
      'Take the maximum of each register',
      'Multiply all register values',
    ],
    correctAnswer: 2,
    explanation:
      'HyperLogLogs are merged by taking the MAX of each register across HLLs. Since each element hashes to a deterministic bucket, the same element will update the same register on different servers. Taking MAX (not sum) ensures each unique element is counted once. This enables distributed counting with minimal data transfer.',
  },
  {
    id: 'mc3',
    question:
      'What is the approximate standard error for HyperLogLog with 16,384 buckets (precision=14)?',
    options: [
      '0.01% (very accurate)',
      '0.81% (typical production)',
      '10% (high error)',
      '50% (unusable)',
    ],
    correctAnswer: 1,
    explanation:
      'Standard error ≈ 1.04 / √m where m = number of buckets. With 16,384 buckets: 1.04 / √16384 = 1.04 / 128 ≈ 0.81%. In practice, this means estimates are within 1-2% of actual count 95% of the time. This accuracy is acceptable for analytics and monitoring use cases.',
  },
  {
    id: 'mc4',
    question: 'Which production systems use HyperLogLog?',
    options: [
      'Only academic research systems',
      'Redis PFCOUNT and Google BigQuery APPROX_COUNT_DISTINCT',
      'Only in-memory databases',
      'Only blockchain systems',
    ],
    correctAnswer: 1,
    explanation:
      'HyperLogLog is production-proven: Redis (PFCOUNT commands), Google BigQuery (APPROX_COUNT_DISTINCT), Facebook (DAU counting), Amazon (CloudWatch metrics), Presto, Apache Druid. Any large-scale analytics system uses HyperLogLog for cardinality estimation. It is the industry standard for approximate unique counting.',
  },
  {
    id: 'mc5',
    question: 'When should you NOT use HyperLogLog?',
    options: [
      'Counting unique website visitors',
      'Estimating distinct search queries',
      'Counting financial transactions requiring exact amounts',
      'Monitoring unique API callers',
    ],
    correctAnswer: 2,
    explanation:
      'Do NOT use HyperLogLog for exact counts required by financial systems, compliance, or anything where 1-2% error is unacceptable. HyperLogLog is perfect for analytics (DAU, unique visitors, monitoring) where approximate counts are sufficient. For financial transactions, use exact counting despite memory cost.',
  },
];
