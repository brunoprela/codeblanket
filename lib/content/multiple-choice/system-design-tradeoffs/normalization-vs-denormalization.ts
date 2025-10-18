/**
 * Multiple choice questions for Normalization vs Denormalization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const normalizationvsdenormalizationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is the main goal of database normalization?',
      options: [
        'Improve query performance',
        'Reduce data redundancy and improve data integrity',
        'Increase storage capacity',
        'Simplify application code',
      ],
      correctAnswer: 1,
      explanation:
        'Database normalization aims to reduce data redundancy and improve data integrity by organizing data into tables with relationships. This ensures each piece of data is stored only once, making updates easier and preventing inconsistencies. However, it may require JOINs which can slow queries.',
    },
    {
      id: 'mc2',
      question: 'When should you consider denormalizing your database?',
      options: [
        'When writes are more frequent than reads',
        'When data integrity is the top priority',
        'When read performance is critical and read:write ratio is high (10:1 or more)',
        'When storage is very limited',
      ],
      correctAnswer: 2,
      explanation:
        'Denormalization makes sense when read performance is critical and you have a high read:write ratio (like 10:1 or higher). The duplicated data speeds up reads by avoiding JOINs, and since writes are infrequent, the added write complexity is acceptable. Examples: product catalogs, social media timelines.',
    },
    {
      id: 'mc3',
      question: 'What is a materialized view?',
      options: [
        'A virtual table that stores no data',
        'A pre-computed query result that is stored and periodically refreshed',
        'A type of index',
        'A database backup',
      ],
      correctAnswer: 1,
      explanation:
        "A materialized view is a pre-computed query result that is stored in the database and periodically refreshed. Unlike regular views (which are virtual), materialized views store actual data, making queries against them very fast. They're useful for complex aggregations that are expensive to compute on every query.",
    },
    {
      id: 'mc4',
      question: 'Why do NoSQL databases often require denormalization?',
      options: [
        'They have unlimited storage',
        "They don't support or have expensive JOIN operations",
        'They are always faster',
        'They never need to update data',
      ],
      correctAnswer: 1,
      explanation:
        "NoSQL databases like MongoDB, Cassandra, and DynamoDB don't support JOIN operations or make them very expensive. To avoid application-level JOINs, you must denormalize by embedding related data in the same document/row. This makes reads fast but requires careful consistency management for updates.",
    },
    {
      id: 'mc5',
      question: 'What is the main trade-off when denormalizing data?',
      options: [
        'Faster reads but more complex writes and storage overhead',
        'Faster writes but slower reads',
        'Less storage but slower queries',
        'Better security but worse performance',
      ],
      correctAnswer: 0,
      explanation:
        'Denormalization trades write complexity and storage for read performance. Reads become faster because data is pre-computed and co-located (no JOINs). But writes become more complex (must update multiple places) and you need more storage (duplicated data). This trade-off makes sense for read-heavy workloads.',
    },
  ];
