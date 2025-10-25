import { MultipleChoiceQuestion } from '@/lib/types';

export const performanceOptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-perf-mc-1',
    question: 'What is the most common cause of slow SQLAlchemy queries?',
    options: [
      'Too many indexes',
      'N+1 query problem',
      'Using PostgreSQL',
      'Too much memory',
    ],
    correctAnswer: 1,
    explanation:
      'N+1 query problem is the most common performance issue: Loading parent records, then querying for each child relationship in a loop. ' +
      'Example: Load 100 users, then query posts for each user = 101 queries (1 + 100). ' +
      'Solution: Use joinedload() or selectinload() to eager load relationships. ' +
      'joinedload uses JOIN (1 query), selectinload uses IN clause (2 queries). ' +
      'Detection: Enable query logging (echo=True), count queries. ' +
      'Fix can improve performance from 5s to 50ms (100x).',
  },
  {
    id: 'sql-perf-mc-2',
    question:
      'When should you use bulk_insert_mappings() instead of session.add()?',
    options: [
      'Never, add() is always better',
      'Only for small datasets (< 10 records)',
      'For inserting large datasets (100+ records)',
      'Only with SQLite',
    ],
    correctAnswer: 2,
    explanation:
      'Use bulk_insert_mappings() for large datasets (100+ records): 10-100x faster than add() in loops. ' +
      'bulk_insert_mappings() bypasses ORM overhead (no instance creation, no flush tracking). ' +
      'Example: session.bulk_insert_mappings(User, [{"name": "User1"}, {"name": "User2"}]). ' +
      'Trade-off: No relationship handling, no events fired, no before_insert triggers. ' +
      'Use add() for small datasets or when you need ORM features (relationships, events). ' +
      'Benchmark: 10,000 inserts - bulk: 1s, add(): 100s.',
  },
  {
    id: 'sql-perf-mc-3',
    question: 'What does EXPLAIN ANALYZE do in PostgreSQL?',
    options: [
      'Optimizes the query automatically',
      'Shows the query execution plan and actual runtime',
      'Creates an index',
      'Caches the query',
    ],
    correctAnswer: 1,
    explanation:
      'EXPLAIN ANALYZE shows query execution plan AND runs the query to get actual runtime. ' +
      'Output: Sequential Scan vs Index Scan, cost estimate, actual time, rows returned. ' +
      'Usage: EXPLAIN ANALYZE SELECT * FROM users WHERE email = "test@example.com". ' +
      'Identifies issues: Sequential scan (missing index), high cost (inefficient join), actual time >> estimated (outdated stats). ' +
      'Fix: Add index to convert Seq Scan to Index Scan (100x faster). ' +
      'EXPLAIN (without ANALYZE) only shows plan, does not execute.',
  },
  {
    id: 'sql-perf-mc-4',
    question: 'What is the benefit of using deferred() column loading?',
    options: [
      'Speeds up inserts',
      'Reduces initial query size by loading large columns only when accessed',
      'Improves index performance',
      'Enables caching',
    ],
    correctAnswer: 1,
    explanation:
      'deferred() loads columns only when accessed (lazy loading for columns). ' +
      'Use for large columns: TEXT, BYTEA, JSON - that are rarely needed. ' +
      'Example: class Article(Base): content = Column(Text, deferred=True). ' +
      'Query: select(Article) loads all columns except content. article.content triggers second query. ' +
      'Benefit: Initial query smaller/faster. Accessing content later is slower (extra query). ' +
      'Use when: Large columns exist but rarely accessed. Example: article listing (no content), detail page (with content).',
  },
  {
    id: 'sql-perf-mc-5',
    question: 'What is a composite index and when should you use it?',
    options: [
      'Index on multiple tables',
      'Index on multiple columns of same table',
      'Index that combines B-tree and Hash',
      'Temporary index',
    ],
    correctAnswer: 1,
    explanation:
      'Composite index: Index on multiple columns of same table. ' +
      'Example: CREATE INDEX idx_user_status_created ON users (status, created_at). ' +
      'Use when: Query filters/sorts on multiple columns: WHERE status = "active" ORDER BY created_at. ' +
      'Column order matters: Most selective column first (status - 3 values, created_at - many values). ' +
      'Supports prefix queries: WHERE status = "active" uses index. WHERE created_at > ... does NOT (wrong order). ' +
      'Benefit: Single composite index faster than multiple single-column indexes.',
  },
];
