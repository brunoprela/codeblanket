/**
 * Multiple choice questions for MongoDB section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const mongodbMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the maximum document size in MongoDB?',
    options: ['1 MB', '4 MB', '16 MB', '100 MB'],
    correctAnswer: 2,
    explanation:
      'MongoDB documents are limited to 16 MB. This limit ensures documents remain reasonable size for performance. For larger data (images, videos), store in GridFS (splits into chunks) or external storage (S3) with reference in MongoDB. Excessive embedding approaching limit signals need for referencing.',
  },
  {
    id: 'mc2',
    question: 'What is the role of the primary node in a MongoDB replica set?',
    options: [
      'Only handles reads',
      'Receives all writes and replicates to secondaries',
      'Stores backup data',
      'Routes queries',
    ],
    correctAnswer: 1,
    explanation:
      'Primary node in replica set receives ALL writes and replicates changes to secondary nodes asynchronously. Secondaries can optionally serve reads (with potential staleness). If primary fails, secondaries automatically elect new primary via majority vote. Only one primary per replica set.',
  },
  {
    id: 'mc3',
    question: 'What does the aggregation framework in MongoDB provide?',
    options: [
      'Data encryption',
      'Pipeline-based data transformation and analysis (like SQL GROUP BY, JOIN)',
      'User authentication',
      'Backup management',
    ],
    correctAnswer: 1,
    explanation:
      'Aggregation framework provides powerful pipeline-based data processing with stages like $match (filter), $group (aggregate), $project (reshape), $lookup (join), $sort, $limit. Similar to SQL aggregations but for documents. Executes server-side for efficiency. Essential for analytics on MongoDB data.',
  },
  {
    id: 'mc4',
    question: 'What is the difference between BSON and JSON?',
    options: [
      'No difference',
      'BSON is binary encoding of JSON with additional types (Date, ObjectId, Binary)',
      'BSON is compressed JSON',
      'BSON is encrypted JSON',
    ],
    correctAnswer: 1,
    explanation:
      "BSON (Binary JSON) is MongoDB's binary storage format. Benefits over JSON: (1) Additional types (Date, ObjectId, Binary, Decimal128), (2) Efficient traversal (includes length prefixes), (3) Fixed-width elements for faster parsing. BSON enables MongoDB's rich type system while maintaining JSON-like flexibility.",
  },
  {
    id: 'mc5',
    question: 'When should you NOT use MongoDB?',
    options: [
      'Need flexible schema',
      'Need complex multi-table joins and transactions',
      'Document-oriented data',
      'Rapid development',
    ],
    correctAnswer: 1,
    explanation:
      'MongoDB is not ideal for: (1) Complex joins (no native join support, use $lookup but limited), (2) Strict relational integrity (use RDBMS), (3) Complex transactions across collections (limited support), (4) Graph traversals (use Neo4j). MongoDB excels at flexible schemas, document storage, and hierarchical data.',
  },
];
