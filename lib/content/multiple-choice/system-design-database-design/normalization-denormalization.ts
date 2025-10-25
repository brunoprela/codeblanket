/**
 * Multiple choice questions for Database Normalization & Denormalization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const normalizationdenormalizationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'norm-1',
      question:
        'Which normal form violation is present in this table?\\n\\nCREATE TABLE employees (\\n  employee_id INT PRIMARY KEY,\\n  employee_name VARCHAR(255),\\n  department_id INT,\\n  department_name VARCHAR(255),\\n  department_location VARCHAR(255)\\n);',
      options: [
        'First Normal Form (1NF) - contains non-atomic values',
        'Second Normal Form (2NF) - partial dependency on composite key',
        'Third Normal Form (3NF) - transitive dependency exists',
        'Boyce-Codd Normal Form (BCNF) - overlapping candidate keys',
      ],
      correctAnswer: 2,
      explanation:
        "Option C is correct. This table violates 3NF due to transitive dependency: department_name and department_location depend on department_id, not directly on the primary key (employee_id). The dependency chain is: employee_id → department_id → department_name/location. To fix, create a separate departments table. It\'s not 1NF violation (all values are atomic). Not 2NF violation (there's no composite primary key). Not BCNF issue (no overlapping candidate keys).",
    },
    {
      id: 'norm-2',
      question: 'When is denormalization most appropriate?',
      options: [
        'When data integrity and consistency are the top priority',
        'When you have a write-heavy workload with frequent updates',
        'When you have a read-heavy workload with known query patterns',
        'When you need to support complex many-to-many relationships',
      ],
      correctAnswer: 2,
      explanation:
        'Option C is correct. Denormalization is most appropriate for read-heavy workloads (e.g., 10:1 or 100:1 read:write ratio) where you know the query patterns and can optimize for them. It reduces JOINs and improves read performance at the cost of data redundancy and write complexity. Option A favors normalization. Option B also favors normalization (frequent updates are easier with normalized data). Option D suggests complex relationships, which are typically better handled with normalization.',
    },
    {
      id: 'norm-3',
      question:
        'What is the main difference between a materialized view and a regular view?',
      options: [
        'Materialized views support more complex queries than regular views',
        'Materialized views store computed results physically, regular views are virtual',
        'Materialized views are always up-to-date, regular views may be stale',
        'Materialized views require less storage than regular views',
      ],
      correctAnswer: 1,
      explanation:
        "Option B is correct. Materialized views physically store the computed query results on disk, allowing fast access without re-running the query. Regular views are virtual - they're just stored query definitions that are executed when accessed. This makes materialized views faster to query but requires periodic refreshing to stay current, and they consume storage. Option A is incorrect (both support similar complexity). Option C is backwards (materialized views may be stale). Option D is incorrect (materialized views require MORE storage).",
    },
    {
      id: 'norm-4',
      question:
        'You have a posts table with a like_count column (denormalized) that is updated every time someone likes/unlikes a post. What is the biggest risk?',
      options: [
        'The like_count column will consume too much storage space',
        'Query performance will be slower than querying the likes table directly',
        'The like_count may become inconsistent if updates fail or race conditions occur',
        'The database will not allow you to create indexes on denormalized columns',
      ],
      correctAnswer: 2,
      explanation:
        'Option C is correct. The biggest risk of denormalization is data inconsistency. If a like is inserted but the like_count update fails (e.g., transaction rollback, application error), or if concurrent updates cause race conditions, the denormalized count may diverge from the true count in the likes table. Option A is incorrect (one integer column is negligible). Option B is incorrect (denormalized count is faster to query). Option D is incorrect (you can index denormalized columns).',
    },
    {
      id: 'norm-5',
      question:
        'In a CQRS (Command Query Responsibility Segregation) pattern, how are the write and read models typically structured?',
      options: [
        'Both write and read models are fully normalized',
        'Both write and read models are fully denormalized',
        'Write model is normalized (source of truth), read model is denormalized (optimized)',
        'Write model is denormalized, read model is normalized',
      ],
      correctAnswer: 2,
      explanation:
        "Option C is correct. CQRS separates write and read models: the write model (command side) is typically normalized to maintain data integrity as the source of truth, while the read model (query side) is denormalized and optimized for specific query patterns. Events from the write model sync to the read model, which may be materialized views, caching layers, or search indexes. This allows ACID writes and fast reads. Options A, B, and D don't capture this separation of concerns.",
    },
  ];
