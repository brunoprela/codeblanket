/**
 * Multiple choice questions for DynamoDB section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dynamodbMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What determines how DynamoDB partitions data?',
    options: [
      'Sort key',
      'Partition key (via hash)',
      'Timestamp',
      'Table name',
    ],
    correctAnswer: 1,
    explanation:
      'DynamoDB partitions data based on the partition key using consistent hashing: hash (partition_key) → partition. Items with the same partition key are stored together. Good partition key design (high cardinality, even distribution) is critical for performance and avoiding hot partitions.',
  },
  {
    id: 'mc2',
    question:
      'What is the difference between on-demand and provisioned capacity mode?',
    options: [
      'No difference, just naming',
      'On-demand: pay per request, scales automatically. Provisioned: pre-specify RCU/WCU',
      'On-demand is always cheaper',
      'Provisioned requires manual scaling',
    ],
    correctAnswer: 1,
    explanation:
      'On-demand: Pay per request ($0.25/million reads, $1.25/million writes), scales automatically to any workload, no capacity planning. Provisioned: Pre-specify RCU/WCU, auto-scaling available, cheaper at steady high volume. Choose on-demand for unpredictable workloads, provisioned for predictable steady-state.',
  },
  {
    id: 'mc3',
    question: 'What is the maximum size of a DynamoDB item?',
    options: ['100 KB', '400 KB', '1 MB', '16 MB'],
    correctAnswer: 1,
    explanation:
      'DynamoDB items are limited to 400 KB including attribute names and values. This includes all attributes for the item. For larger objects, store metadata in DynamoDB and actual data in S3. The partition key and sort key combined also count toward this limit.',
  },
  {
    id: 'mc4',
    question: 'How many items can you write in a single BatchWriteItem call?',
    options: ['10', '25', '100', 'Unlimited'],
    correctAnswer: 1,
    explanation:
      'BatchWriteItem allows up to 25 PutItem or DeleteItem operations in a single call (up to 16 MB total). Provides better throughput than individual writes. Partial failures possible - unprocessed items returned for retry. Also useful for batch deletes.',
  },
  {
    id: 'mc5',
    question: 'What is DynamoDB Streams used for?',
    options: [
      'Backup only',
      'Capture item-level changes for CDC (change data capture)',
      'Query optimization',
      'Data compression',
    ],
    correctAnswer: 1,
    explanation:
      'DynamoDB Streams captures item-level changes (inserts, updates, deletes) in time-ordered sequence. Use cases: (1) Trigger Lambda functions on changes, (2) Replicate to other systems, (3) Audit logging, (4) Materialized views, (5) Real-time notifications. Streams retain records for 24 hours.',
  },
];
