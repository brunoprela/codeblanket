export const performanceOptimization = {
  title: 'Performance Optimization',
  id: 'performance-optimization',
  content: `
# Performance Optimization

## Introduction

Performance optimization separates basic applications from production-grade systems. This section covers profiling, query optimization, indexing strategies, caching, and connection pooling for high-performance SQLAlchemy applications.

**Core topics:**
- Query profiling and EXPLAIN analysis
- Index strategies and optimization
- Bulk operations (100x faster)
- Query result caching
- Connection pooling tuning
- Memory management
- N+1 problem solutions
- Production monitoring

---

## Query Profiling

### Using EXPLAIN

\`\`\`python
# Analyze query performance
stmt = select(User).where(User.email == "test@example.com")

# Get compiled SQL
print(stmt.compile(compile_kwargs={"literal_binds": True}))

# EXPLAIN (PostgreSQL)
from sqlalchemy import text
explain = text(f"EXPLAIN ANALYZE {stmt}")
result = session.execute(explain)
for row in result:
    print(row)

# Look for: Seq Scan (bad), Index Scan (good), execution time
\`\`\`

### Query Optimization Checklist

✅ Add indexes to WHERE clause columns  
✅ Use select...

(continuing in next message to stay within limits)
\`\`\`

---

## Bulk Operations

\`\`\`python
# BAD: Individual inserts (slow)
for i in range(10000):
    user = User(email=f"user{i}@example.com")
    session.add(user)
session.commit()  # 10+ seconds

# GOOD: Bulk insert mappings (100x faster)
users = [{"email": f"user{i}@example.com"} for i in range(10000)]
session.bulk_insert_mappings(User, users)
session.commit()  # 0.1 seconds
\`\`\`

---

## Summary

✅ Profile with EXPLAIN ANALYZE  
✅ Index WHERE/JOIN columns  
✅ Use bulk operations for large datasets  
✅ Cache expensive queries  
✅ Tune connection pool size  
✅ Avoid N+1 with eager loading  
✅ Monitor query performance in production
`,
};

export const performanceOptimizationQuiz = [
  {
    id: 'sql-perf-q-1',
    question:
      'You have a query taking 5 seconds on 1M users. Explain how to optimize it using EXPLAIN ANALYZE, indexing, and query restructuring.',
    sampleAnswer:
      'Performance optimization process: (1) Profile with EXPLAIN ANALYZE - identifies sequential scans. (2) Add indexes to WHERE clause columns. (3) Use selectinload to avoid N+1. (4) Verify with EXPLAIN - should show Index Scan. Result: 5s → 50ms. Monitor with APM tools in production.',
    keyPoints: [
      'EXPLAIN ANALYZE finds bottlenecks',
      'Index WHERE/JOIN columns',
      'Eager load relationships',
      'Verify optimization worked',
      'Monitor in production',
    ],
  },
  {
    id: 'sql-perf-q-2',
    question:
      'Compare bulk_insert_mappings vs individual session.add() for inserting 100K records. Include performance numbers and when to use each.',
    sampleAnswer:
      'Bulk operations comparison: session.add() in loop: 100K records = 300 seconds (1 INSERT per record, transaction overhead). bulk_insert_mappings: 100K records = 3 seconds (batched INSERTs, minimal overhead). 100x faster. Use bulk for: data imports, migrations, batch processing. Use session.add for: user-facing operations, when need object state, relationship handling. Bulk limitations: No relationship handling, no Python defaults, no events triggered.',
    keyPoints: [
      'bulk_insert_mappings 100x faster than add() loop',
      'Bulk: 100K records in 3s vs 300s',
      'Use bulk for: imports, migrations, batch',
      'Use add for: relationships, events, object state',
      'Bulk limitations: no relationships/events',
    ],
  },
  {
    id: 'sql-perf-q-3',
    question:
      'Design a caching strategy for a high-traffic API endpoint that loads user profiles. Include cache invalidation and performance metrics.',
    sampleAnswer:
      'Profile caching strategy: (1) Layer 1 - Redis cache: Cache full user objects, key="user:{id}", TTL=1 hour. (2) Cache-aside pattern: Check cache first, on miss query DB and cache result. (3) Invalidation: On user update, delete cache key. Use Redis pub/sub for distributed invalidation. (4) Performance: Cache hit: 1ms, Cache miss: 50ms DB query + cache write. 95% hit rate = 95% of requests served in 1ms. (5) Monitoring: Track hit rate, cache size, TTL effectiveness. Alert if hit rate < 80%.',
    keyPoints: [
      'Redis cache-aside: check cache → miss → query DB → cache',
      'TTL 1 hour, invalidate on updates',
      'Cache hit: 1ms vs DB: 50ms',
      '95% hit rate = 95% requests in 1ms',
      'Monitor: hit rate, size, alerts',
    ],
  },
];

export const performanceOptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-perf-mc-1',
    question: 'What does "Seq Scan" in EXPLAIN output indicate?',
    options: [
      'Efficient query',
      'Sequential scan - missing index, slow for large tables',
      'Optimal performance',
      'Cache hit',
    ],
    correctAnswer: 1,
    explanation:
      'Seq Scan (Sequential Scan) means database scans entire table row-by-row. Slow for large tables. Indicates missing index on WHERE clause columns. Solution: CREATE INDEX on queried columns. After indexing, EXPLAIN should show "Index Scan" instead.',
  },
  {
    id: 'sql-perf-mc-2',
    question: 'How much faster is bulk_insert_mappings vs individual inserts?',
    options: ['Same speed', '2x faster', '10x faster', '100x faster'],
    correctAnswer: 3,
    explanation:
      'bulk_insert_mappings is typically 100x faster than individual session.add() calls. For 10,000 inserts: individual = 30s, bulk = 0.3s. Bulk batches INSERTs and minimizes transaction overhead. Use for: data imports, migrations, batch processing. Note: bulk operations bypass relationships and events.',
  },
  {
    id: 'sql-perf-mc-3',
    question:
      'What is the recommended pool_size for a web application handling 1000 concurrent requests?',
    options: [
      '1000 (one per request)',
      '5-20 per app instance',
      '1 (shared connection)',
      '100 per instance',
    ],
    correctAnswer: 1,
    explanation:
      'Pool size should be 5-20 per application instance, NOT one per request. With connection pooling, requests reuse connections. 1000 concurrent requests across 10 app instances = 100 concurrent/instance. pool_size=20 + max_overflow=30 = 50 max handles this. Total connections across all instances must not exceed database max_connections (PostgreSQL default: 500).',
  },
  {
    id: 'sql-perf-mc-4',
    question: 'What is the primary benefit of query result caching?',
    options: [
      'Reduces code complexity',
      'Avoids database queries for repeated data, dramatically improves response time',
      'Improves data accuracy',
      'Simplifies debugging',
    ],
    correctAnswer: 1,
    explanation:
      'Caching avoids expensive database queries for frequently accessed data. Cache hit: ~1ms, Database query: ~50ms+. For endpoints with 95% cache hit rate, 95% of requests serve in 1ms vs 50ms. Critical for high-traffic endpoints. Trade-off: Must handle cache invalidation when data changes. Common: Redis cache with TTL + manual invalidation on updates.',
  },
  {
    id: 'sql-perf-mc-5',
    question: 'Why should you avoid querying in loops?',
    options: [
      'Code looks messy',
      'Causes N+1 query problem - linear performance degradation',
      'Not allowed by SQLAlchemy',
      'Database security issue',
    ],
    correctAnswer: 1,
    explanation:
      'Queries in loops cause N+1 problem: 1 query for parent objects, N queries for children (one per iteration). Example: 100 users = 101 queries, 10,000 users = 10,001 queries. Solution: Eager loading with selectinload executes only 2 queries regardless of N. N+1 is the most common ORM performance issue. Always use eager loading when accessing relationships in loops.',
  },
];
