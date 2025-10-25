import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncDatabaseOperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ado-mc-1',
    question:
      'Why is asyncpg significantly faster than psycopg2 (blocking driver)?',
    options: [
      'asyncpg is written in C',
      'asyncpg allows concurrent queries and uses binary protocol',
      'asyncpg caches all queries',
      'asyncpg only works with PostgreSQL 15+',
    ],
    correctAnswer: 1,
    explanation:
      'asyncpg is faster because: (1) Concurrent queries: Can execute 100 queries concurrently (psycopg2 sequential). Benchmark: 100 queries × 10ms = 1s sequential vs 10ms concurrent (100× faster). (2) Binary protocol: Uses PostgreSQL binary format (less parsing overhead). (3) Cython implementation: Compiled C extension (faster than pure Python). (4) Connection pooling: Reuses connections efficiently. asyncpg typically 3-5× faster than psycopg2 for single queries, 100× faster for concurrent queries. Works with all PostgreSQL versions 9.5+.',
  },
  {
    id: 'ado-mc-2',
    question: 'What is the purpose of a database connection pool?',
    options: [
      'To cache query results',
      'To reuse database connections, avoiding expensive connection setup',
      'To encrypt database traffic',
      'To automatically retry failed queries',
    ],
    correctAnswer: 1,
    explanation:
      'Connection pool maintains a set of reusable database connections. Creating a new connection is expensive: TCP handshake (50ms) + SSL (50ms) + authentication (20ms) = 120ms overhead. With pool: Reuse existing connection (~0ms overhead). Example: 1000 requests with new connections = 120 seconds wasted. With pool = ~0 seconds. Pool also: Limits total connections (prevents overwhelming database), Warms connections at startup (min_size), Scales on demand (up to max_size). Typical config: min_size=10, max_size=20. Critical for production performance.',
  },
  {
    id: 'ado-mc-3',
    question:
      'What happens when you use async with conn.transaction() in asyncpg?',
    options: [
      'It creates a new database connection',
      'It starts a transaction that auto-commits on success or rolls back on exception',
      'It caches all queries in the transaction',
      'It locks the entire database',
    ],
    correctAnswer: 1,
    explanation:
      'async with conn.transaction() provides automatic transaction management: On enter: Executes BEGIN. During block: All operations part of same transaction. On success: Executes COMMIT (changes saved). On exception: Executes ROLLBACK (changes discarded). Example: async with conn.transaction(): await conn.execute("INSERT ..."); await conn.execute("UPDATE ..."); # Both commit together if no exception, both rollback if exception. Benefits: Atomic operations, automatic rollback, exception-safe. Never need manual BEGIN/COMMIT/ROLLBACK.',
  },
  {
    id: 'ado-mc-4',
    question:
      'Why is copy_records_to_table much faster than executemany for bulk inserts?',
    options: [
      'It uses more connections',
      'It uses PostgreSQL COPY protocol with binary format and single round-trip',
      'It automatically creates indexes',
      'It compresses the data',
    ],
    correctAnswer: 1,
    explanation:
      'copy_records_to_table uses PostgreSQL COPY protocol: Binary format (less parsing overhead than SQL text), Single round-trip (sends all rows at once), Bypasses query planner (direct table insertion). executemany: Text format (must parse SQL), N round-trips (network latency × N), Goes through query planner each time. Benchmark: 1M rows: executemany ~60 seconds, copy_records_to_table ~0.6 seconds (100× faster!). Trade-off: copy_records_to_table less flexible (no complex logic), executemany works with any SQL. Use copy for simple bulk inserts (>1000 rows), executemany for complex queries.',
  },
  {
    id: 'ado-mc-5',
    question:
      'What does pool.acquire() do and why is it important to release connections?',
    options: [
      'It creates a new database',
      'It gets a connection from the pool; must release or pool exhausts',
      'It acquires a database lock',
      'It starts a transaction',
    ],
    correctAnswer: 1,
    explanation:
      'pool.acquire() gets a connection from the pool (or waits if all busy). Must be released back to pool after use: async with pool.acquire() as conn: use conn (auto-releases when exiting). Without release: Connection not returned to pool. Eventually all max_size connections taken. New requests block forever (deadlock). Example: for i in range(100): conn = await pool.acquire(); await conn.fetch(...); # FORGOT to release! After max_size iterations, pool exhausted. Always use async with for automatic release. Or manual: conn = await pool.acquire(); try: use conn; finally: await pool.release(conn).',
  },
];
