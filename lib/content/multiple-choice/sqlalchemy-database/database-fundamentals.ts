import { MultipleChoiceQuestion } from '@/lib/types';

export const databaseFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-db-fund-mc-1',
    question:
      'What is the correct behavior of ON DELETE CASCADE in a foreign key relationship?',
    options: [
      'Prevents deletion of the parent row if child rows exist',
      'Automatically deletes child rows when the parent row is deleted',
      'Sets the foreign key column to NULL when parent is deleted',
      'Creates a backup before deleting',
    ],
    correctAnswer: 1,
    explanation:
      "ON DELETE CASCADE automatically deletes all child rows that reference the parent row when the parent is deleted. This maintains referential integrity by removing orphaned records. Example: Deleting a user with CASCADE on posts.user_id will delete all of that user's posts. Use RESTRICT to prevent deletion, SET NULL to keep children but nullify the relationship.",
  },
  {
    id: 'sql-db-fund-mc-2',
    question:
      'Which Python database driver should you choose for high-performance async PostgreSQL access?',
    options: ['psycopg2', 'pymysql', 'asyncpg', 'sqlite3'],
    correctAnswer: 2,
    explanation:
      'asyncpg is a high-performance async PostgreSQL driver designed for asyncio applications. It is faster than psycopg2 and provides async/await support. psycopg2 is synchronous (though psycopg3 has async support). pymysql is for MySQL, not PostgreSQL. sqlite3 is for SQLite. For production async Python applications with PostgreSQL, asyncpg is the best choice.',
  },
  {
    id: 'sql-db-fund-mc-3',
    question: 'What is the primary advantage of using a connection pool?',
    options: [
      'Encrypts database connections',
      'Reuses connections to avoid expensive connection creation',
      'Automatically backs up data',
      'Provides automatic failover',
    ],
    correctAnswer: 1,
    explanation:
      'Connection pools reuse database connections instead of creating new ones for each request. Creating a new connection is expensive (100-1000ms: TCP handshake, authentication, session setup). Connection pools maintain a pool of open connections (< 1ms to acquire). This dramatically improves performance and prevents connection exhaustion under high load. Critical for production applications.',
  },
  {
    id: 'sql-db-fund-mc-4',
    question:
      'In the context of database transactions, what does the "I" in ACID stand for and what does it guarantee?',
    options: [
      'Integrity - data must be valid',
      "Isolation - concurrent transactions don't interfere",
      'Immutability - data cannot be changed',
      'Identity - each record has unique ID',
    ],
    correctAnswer: 1,
    explanation:
      "Isolation ensures that concurrent transactions don't interfere with each other. Multiple transactions can run simultaneously, but each appears to execute in isolation. Different isolation levels (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE) provide different guarantees about what data transactions can see. Higher isolation prevents dirty reads, non-repeatable reads, and phantom reads, but may reduce concurrency.",
  },
  {
    id: 'sql-db-fund-mc-5',
    question: 'When comparing ORMs, which statement is most accurate?',
    options: [
      'ORMs always perform better than raw SQL',
      'SQLAlchemy is tied to the Django framework',
      'ORMs provide SQL injection protection and database abstraction',
      'Raw SQL should always be used instead of ORMs',
    ],
    correctAnswer: 2,
    explanation:
      'ORMs provide SQL injection protection (parameterized queries) and database abstraction (switch databases with minimal code changes). However, ORMs do NOT always perform better than raw SQL—they have 5-20% overhead for simple queries and raw SQL is much faster for bulk operations. SQLAlchemy is framework-agnostic (Django ORM is tied to Django). The best approach is hybrid: use ORM for productivity (CRUD, simple queries) and raw SQL for performance (analytics, bulk operations).',
  },
];
