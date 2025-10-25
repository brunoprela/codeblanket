import { MultipleChoiceQuestion } from '@/lib/types';

export const performanceOptimizationQuiz = [
  {
    id: 'sql-perf-q-1',
    question:
      'Design a comprehensive performance optimization strategy for a slow SQLAlchemy application. Address: (1) identifying bottlenecks, (2) query optimization techniques, (3) indexing strategy, (4) batch operations, (5) caching. Include code examples and metrics.',
    sampleAnswer:
      'Performance optimization strategy: (1) Identify bottlenecks: Use query logging with echo=True and measure slow queries (>100ms). Enable PostgreSQL slow query log. Profile with py-spy or cProfile. Use EXPLAIN ANALYZE to find sequential scans. (2) Query optimization: Eliminate N+1 queries with joinedload/selectinload. Use select() instead of Query API (faster). Defer loading of large text/binary columns with deferred(). Avoid implicit joins - use explicit join(). (3) Indexing: Create indexes on foreign keys, WHERE clause columns, ORDER BY columns. Use composite indexes for multi-column queries. Partial indexes for filtered queries. Monitor with pg_stat_user_indexes. (4) Batch operations: Use bulk_insert_mappings() instead of add() in loops (100x faster). Batch updates with update() statement. Delete in batches to avoid lock timeouts. (5) Caching: Redis for query results (5min TTL). Use query.options(FromCache("default", cache_key)). Invalidate on writes. Result: Query time reduced from 2s to 50ms, throughput increased 10x.',
    keyPoints: [
      'Profile first: query logging, EXPLAIN ANALYZE, identify N+1 queries',
      'Query optimization: joinedload, select() API, defer large columns',
      'Indexes: foreign keys, WHERE/ORDER BY columns, composite for multi-column',
      'Batch operations: bulk_insert_mappings() 100x faster than loops',
      'Caching: Redis with TTL and explicit invalidation on writes',
    ],
  },
  {
    id: 'sql-perf-q-2',
    question:
      'Explain how to use database indexes effectively in SQLAlchemy. Include: (1) types of indexes (B-tree, Hash, GiST), (2) when to create indexes, (3) composite indexes, (4) partial indexes, (5) monitoring index usage. Provide implementation examples.',
    sampleAnswer:
      'Database indexing strategy: (1) Types: B-tree (default, most common) for =, <, >, ORDER BY. Hash for = only (rare). GiST for full-text search, geometric data. GIN for JSONB, arrays. Example: Index(User.email) creates B-tree. (2) When to create: Foreign keys (always - used in joins). WHERE clause columns (user_id in WHERE user_id = 123). ORDER BY/GROUP BY columns. Unique constraints. NOT on low-cardinality columns (gender - only 2 values). (3) Composite indexes: For multi-column queries. CREATE INDEX idx_user_created ON users (status, created_at). Order matters: most selective column first. Use for queries with multiple WHERE conditions or WHERE + ORDER BY. (4) Partial indexes: For filtered queries. Index(User.email, postgresql_where=User.active==True). Smaller, faster. Use for queries always filtering on same condition. (5) Monitoring: Query pg_stat_user_indexes for index usage. Check idx_scan (index scans) and idx_tup_fetch (rows returned). Drop unused indexes (idx_scan = 0). Verify EXPLAIN uses index (Index Scan not Seq Scan). Result: Queries using indexes are 100-1000x faster.',
    keyPoints: [
      'B-tree default: =, <, >, ORDER BY. GIN for JSONB/arrays',
      'Index foreign keys, WHERE/ORDER BY columns, NOT low-cardinality',
      'Composite: multi-column queries, most selective column first',
      'Partial: smaller/faster for queries with consistent filter',
      'Monitor: pg_stat_user_indexes, drop unused (idx_scan=0)',
    ],
  },
  {
    id: 'sql-perf-q-3',
    question:
      'You have a query that performs poorly at scale. Walk through the process of: (1) identifying the issue with EXPLAIN ANALYZE, (2) fixing with query optimization, (3) adding appropriate indexes, (4) verifying improvement. Include complete examples.',
    sampleAnswer:
      'Query optimization process: (1) Identify issue: Run EXPLAIN ANALYZE. Example: EXPLAIN ANALYZE SELECT * FROM posts WHERE user_id = 123 ORDER BY created_at DESC LIMIT 10. Output shows "Seq Scan on posts (cost=1000..5000)" and "execution time: 2000ms". Problem: Sequential scan (reads all rows). (2) Query optimization: Check for N+1 queries with selectinload(). Avoid SELECT * - fetch only needed columns. Use pagination (limit/offset) to reduce data transfer. Rewrite: select(Post.id, Post.title).where(Post.user_id == 123).order_by(Post.created_at.desc()).limit(10). (3) Add indexes: CREATE INDEX idx_posts_user_created ON posts (user_id, created_at DESC). Composite index supports both WHERE and ORDER BY. Verify: Run EXPLAIN ANALYZE again. Now shows "Index Scan using idx_posts_user_created (cost=0..50)" and "execution time: 5ms". 400x improvement! (4) Verify: Load test with 1000 concurrent requests. Before: 2000ms p95. After: 10ms p95. Monitor query plan remains stable with production data. Result: Query optimized from 2s to 5ms.',
    keyPoints: [
      'EXPLAIN ANALYZE: Identifies Seq Scan, shows cost and execution time',
      'Optimize query: selectinload (no N+1), limit columns, pagination',
      'Composite index: Supports WHERE user_id + ORDER BY created_at',
      'Verify: Re-run EXPLAIN shows Index Scan, measure time improvement',
      'Load test: Verify performance at scale, monitor production',
    ],
  },
];
