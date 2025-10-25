import { MultipleChoiceQuestion } from '@/lib/types';

export const relationshipLoadingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-loading-mc-1',
    question: 'What is the N+1 query problem?',
    options: [
      'A query that takes N+1 seconds to execute',
      '1 query for parent objects, then N additional queries for related objects',
      'N queries that should be combined into 1',
      'A database with N+1 tables',
    ],
    correctAnswer: 1,
    explanation:
      'The N+1 problem occurs when you execute 1 query to load N parent objects, then N additional queries to load related objects for each parent. Example: SELECT * FROM users LIMIT 100 (1 query), then SELECT * FROM posts WHERE user_id = ? for each user (100 queries) = 101 total queries. Solution: Eager loading with selectinload executes just 2 queries total. N+1 causes linear performance degradation: 100 users = 101 queries, 10,000 users = 10,001 queries.',
  },
  {
    id: 'sql-loading-mc-2',
    question:
      'Which loading strategy should be your default for one-to-many relationships?',
    options: ['lazy (default)', 'joinedload', 'selectinload', 'subqueryload'],
    correctAnswer: 2,
    explanation:
      'selectinload is the recommended default for one-to-many relationships. It executes 2 queries: one for parents, one SELECT IN for children. Benefits: No cartesian product (memory efficient), works correctly with LIMIT, scales well for large collections, can load multiple relationships efficiently. joinedload causes cartesian product (user duplicated per post). subqueryload is legacy (selectinload is better). lazy causes N+1 problem.',
  },
  {
    id: 'sql-loading-mc-3',
    question: 'What happens when you use joinedload with LIMIT?',
    options: [
      'Works correctly, limits parent objects',
      'LIMIT applies after JOIN, potentially returning fewer parents than expected',
      'Raises an error',
      'Automatically converts to selectinload',
    ],
    correctAnswer: 1,
    explanation:
      'joinedload + LIMIT is a common bug. LIMIT applies AFTER the JOIN, limiting rows not parent objects. Example: select(User).options(joinedload(User.posts)).limit(10) with first user having 10 posts returns 10 ROWS (1 user with 10 posts), not 10 users. This is almost never the desired behavior. Solution: Always use selectinload with LIMIT. selectinload limits users first (correct), then loads posts for those users.',
  },
  {
    id: 'sql-loading-mc-4',
    question:
      'When should you use contains_eager instead of selectinload or joinedload?',
    options: [
      'Always, it is faster',
      'When you have already explicitly joined the tables for filtering',
      'For many-to-many relationships only',
      'When loading more than 3 relationships',
    ],
    correctAnswer: 1,
    explanation:
      'Use contains_eager when you have explicitly joined tables (usually for filtering) and want to reuse that JOIN to populate the relationship instead of issuing a separate query. Example: select(User).join(User.posts).where(Post.published == True).options(contains_eager(User.posts)) - the explicit JOIN is already there for filtering, contains_eager reuses it. Without contains_eager, SQLAlchemy would load posts again with a separate query (inefficient).',
  },
  {
    id: 'sql-loading-mc-5',
    question:
      'Why does joinedload require calling unique() on results while selectinload does not?',
    options: [
      'It is a SQL Alchemy bug',
      'joinedload creates duplicate parent objects due to cartesian product from JOIN',
      'unique() improves performance',
      'selectinload automatically calls unique()',
    ],
    correctAnswer: 1,
    explanation:
      'joinedload performs a LEFT OUTER JOIN, creating a cartesian product where each parent appears once per child. Example: User with 10 posts returns 10 rows, all with the same User data. Without unique(), you get 10 duplicate User objects in your results list. unique() deduplicates them. selectinload uses separate queries (no JOIN), so no cartesian product and no duplicates. Code: joinedload: session.execute(stmt).scalars().unique().all(). selectinload: session.execute(stmt).scalars().all() - no unique() needed.',
  },
];
