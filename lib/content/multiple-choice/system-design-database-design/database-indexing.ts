/**
 * Multiple choice questions for Database Indexing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const databaseindexingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'indexing-1',
    question:
      'You have a query: SELECT * FROM orders WHERE user_id = 123 AND status = "active" AND created_at > "2024-01-01". Which composite index would be MOST efficient?',
    options: [
      'CREATE INDEX idx ON orders(created_at, user_id, status)',
      'CREATE INDEX idx ON orders(user_id, status, created_at)',
      'CREATE INDEX idx ON orders(status, user_id, created_at)',
      'CREATE INDEX idx ON orders(created_at, status, user_id)',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. For composite indexes, order matters: (1) Equality filters first (user_id, status), (2) Range filters last (created_at). This follows the left-prefix rule and allows the database to efficiently filter by user_id and status, then scan the remaining rows by created_at. Option A puts the range filter first, making user_id and status filters inefficient. Option C starts with status (low selectivity). Option D also has the range filter early.',
  },
  {
    id: 'indexing-2',
    question: 'Which statement about index trade-offs is INCORRECT?',
    options: [
      'Each additional index increases storage requirements',
      'More indexes always improve query performance',
      'Indexes slow down INSERT, UPDATE, and DELETE operations',
      'Unused indexes should be removed to improve write performance',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is incorrect. More indexes do NOT always improve performance. Too many indexes hurt write performance (every write must update all indexes), increase storage costs, and can confuse the query optimizer. Indexes should be created strategically based on query patterns. Options A, C, and D are all correct statements about index trade-offs.',
  },
  {
    id: 'indexing-3',
    question:
      'Why might a database optimizer choose a full table scan over using an available index?',
    options: [
      'The index is corrupted and needs to be rebuilt',
      'The table is too large to fit in memory',
      'The query matches too many rows (low selectivity)',
      'The index was created recently and is not yet available',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. When a query matches a large percentage of rows (e.g., 20-30%+), the cost of random I/O from index lookups exceeds the cost of a sequential table scan. The optimizer chooses the more efficient full scan. Example: "WHERE country = \'USA\'" in a US-based company might match 90% of rows. Option A would cause errors, not fall back to table scan. Option B is not a reason to avoid indexes. Option D is incorrect; indexes are immediately available after creation.',
    difficulty: 'hard' as const,
  },
  {
    id: 'indexing-4',
    question: 'What is a covering index and when is it beneficial?',
    options: [
      'An index that covers all columns in the table for maximum performance',
      'An index that includes all columns needed by a query, avoiding table lookups',
      'An index that covers fragmented data to improve storage efficiency',
      'An index that is automatically created to cover foreign key relationships',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. A covering index (or index with INCLUDE columns) contains all data needed to satisfy a query, allowing an "index-only scan" without accessing table rows. This eliminates random I/O to fetch table data. Example: CREATE INDEX idx ON orders(user_id) INCLUDE (status, total) can satisfy "SELECT status, total FROM orders WHERE user_id = 123" entirely from the index. Option A is impractical (huge index). Option C misunderstands covering. Option D describes a different concept.',
    difficulty: 'medium' as const,
  },
  {
    id: 'indexing-5',
    question:
      'You have a users table with 1 billion rows. Only 0.1% of users are "premium" status. What indexing strategy would optimize queries for premium users?',
    options: [
      'CREATE INDEX idx ON users(status)',
      'CREATE INDEX idx ON users(user_id, status)',
      'CREATE INDEX idx ON users(status, user_id) WHERE status = "premium"',
      'CREATE UNIQUE INDEX idx ON users(user_id) WHERE status = "premium"',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. A partial (filtered) index on WHERE status = "premium" creates a tiny index covering only 1M rows (0.1%) instead of 1B rows. This makes the index 1000x smaller, faster to search, and much cheaper to maintain on writes. The composite (status, user_id) supports various queries. Option A creates a huge index for low-selectivity column. Option B doesn\'t leverage the small subset. Option D is incorrect because user_id should already be unique across all users, not just premium users.',
    difficulty: 'hard' as const,
  },
];
