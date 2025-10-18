/**
 * Quiz questions for Database Indexing section
 */

export const databaseindexingQuiz = [
  {
    id: 'indexing-disc-1',
    question:
      'Design an indexing strategy for a high-traffic e-commerce platform with 100 million products. Consider common query patterns: browse by category, search by keywords, filter by price range, and sort by popularity. Discuss trade-offs between read performance, write performance, and storage.',
    sampleAnswer: `Comprehensive indexing strategy for e-commerce platform:

**Table Structure:**
\`\`\`sql
CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    popularity_score INTEGER,
    stock_quantity INTEGER,
    created_at TIMESTAMP,
    search_vector TSVECTOR,
    metadata JSONB
);
\`\`\`

**Index Strategy:**

1. **Category Browsing:**
\`\`\`sql
CREATE INDEX idx_category_popularity ON products(category, popularity_score DESC)
WHERE stock_quantity > 0;
\`\`\`
- Supports: "Show popular products in category"
- Partial index (in-stock only) reduces size by ~20%
- Composite order: category (filter) then popularity (sort)

2. **Price Filtering:**
\`\`\`sql
CREATE INDEX idx_category_price ON products(category, price)
WHERE stock_quantity > 0;
\`\`\`
- Supports: "Products in category under $50"
- Partial index for in-stock products

3. **Full-Text Search:**
\`\`\`sql
CREATE INDEX idx_search ON products USING GIN(search_vector);
\`\`\`
- Supports: Keyword search with ranking
- GIN index: ~15-20% of table size

4. **Recently Added:**
\`\`\`sql
CREATE INDEX idx_new_products ON products(created_at DESC)
WHERE created_at > NOW() - INTERVAL '30 days';
\`\`\`
- Supports: "New arrivals" page
- Partial index: only recent products (much smaller)

**Trade-offs Analysis:**

*Storage Impact:*
- Base table: 50GB (100M products × ~500 bytes)
- Indexes: ~30GB total (60% overhead)
- Total: 80GB

*Write Performance:*
- Without indexes: 1000 products/sec
- With indexes: 400 products/sec (60% reduction)
- Mitigation: Batch inserts, async index updates

*Read Performance:*
- Category browse: 1000ms → 10ms (100x faster)
- Search: 5000ms → 50ms (100x faster)
- Price filter: 2000ms → 20ms (100x faster)

**Optimization Strategies:**

1. **Partition by Category:** Split hot categories into separate partitions with independent indexes
2. **Materialized Views:** Pre-compute popular queries (e.g., "trending products")
3. **Cache Layer:** Redis for top 1% of products (99% of traffic)
4. **Index Maintenance:** Weekly REINDEX during low-traffic windows
5. **Monitor Usage:** Drop indexes with idx_scan < 1000/week

**Advanced Considerations:**
- Separate search cluster (Elasticsearch) for complex text queries
- Read replicas with different index strategies for analytics vs transactional queries
- Archive old products to historical partition with fewer indexes`,
    keyPoints: [
      'Composite indexes match query patterns (category + popularity)',
      'Partial indexes reduce size for common filters (in-stock only)',
      'Specialized indexes for different use cases (GIN for full-text)',
      'Trade-offs: 60% storage overhead, 60% write slowdown, 100x read speedup',
      'Monitor and drop unused indexes to optimize performance',
    ],
  },
  {
    id: 'indexing-disc-2',
    question:
      'A social media application has a posts table with 10 billion rows. The main query is: "Get recent posts from users I follow, sorted by time." Discuss how you would design indexes to make this query efficient while managing the massive scale. Consider sharding, partitioning, and alternative approaches.',
    sampleAnswer: `Comprehensive solution for social media feed at scale:

**Challenge Analysis:**
- 10B posts: single index would be massive (>500GB)
- Query: WHERE user_id IN (following_list) ORDER BY created_at DESC
- Requirements: <100ms p99 latency, high write throughput

**Solution 1: Sharding by User**

\`\`\`sql
-- Shard posts by user_id (e.g., 100 shards)
-- Each shard: ~100M posts

-- Per-shard index
CREATE INDEX idx_posts_timeline ON posts_shard_N(user_id, created_at DESC);
-- Each index: ~5GB instead of 500GB
\`\`\`

*Pros:*
- Smaller indexes per shard (faster, less memory)
- Write throughput scales linearly
- Easier maintenance (reindex one shard at a time)

*Cons:*
- Feed query must scatter-gather across user shards
- Complex: if following 100 users, query 100 shards
- Merge sort results from multiple shards

**Solution 2: Time-Based Partitioning**

\`\`\`sql
-- Partition by month
CREATE TABLE posts_2024_01 PARTITION OF posts
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Composite index per partition
CREATE INDEX idx_posts_2024_01_feed 
ON posts_2024_01(user_id, created_at DESC);
\`\`\`

*Pros:*
- Recent data (hot partitions) have small, fast indexes
- Old partitions can use compressed storage
- Query only recent partitions (last 7 days)

*Cons:*
- Still need to query all followed users
- Less effective for users who post infrequently

**Solution 3: Pre-computed Fan-out (Recommended)**

Instead of query-time JOIN, precompute the feed:

\`\`\`sql
-- Fan-out on write: when user posts, write to all followers' feeds
CREATE TABLE user_feeds (
    user_id BIGINT,          -- Feed owner
    post_id BIGINT,
    author_id BIGINT,
    created_at TIMESTAMP
);

-- Simple index for feed retrieval
CREATE INDEX idx_user_feed ON user_feeds(user_id, created_at DESC);
\`\`\`

*Read Path (Fast):*
\`\`\`sql
-- Simple query: just read your feed
SELECT * FROM user_feeds
WHERE user_id = 123
ORDER BY created_at DESC
LIMIT 50;
\`\`\`

*Write Path (Complex):*
\`\`\`
When user posts:
1. Insert into posts table
2. Fan-out: Insert post_id into all followers' feeds
3. If celebrity (1M followers): async background job
\`\`\`

*Hybrid Approach:*
- Regular users (<1000 followers): fan-out on write
- Celebrities: fan-out on read (mix live data at query time)

**Solution 4: Covering Index + Caching**

\`\`\`sql
-- Covering index avoids table lookups
CREATE INDEX idx_feed_covering ON posts(user_id, created_at DESC)
INCLUDE (content, media_urls, like_count);

-- Shard by user_id hash
-- Cache recent posts in Redis per user
\`\`\`

*Caching Strategy:*
\`\`\`
Redis Cache:
- Key: "feed:user:123"
- Value: Sorted set of recent post IDs (500 most recent)
- TTL: 1 hour

On cache miss:
- Query database
- Populate cache
- Async refresh
\`\`\`

**Comparative Analysis:**

| Approach | Read Latency | Write Latency | Storage | Complexity |
|----------|-------------|---------------|---------|------------|
| Sharding | 200ms | 10ms | 1x | High |
| Partitioning | 150ms | 10ms | 1x | Medium |
| Fan-out | 20ms | 100ms | 3x | High |
| Covering + Cache | 50ms | 10ms | 1.5x | Medium |

**Recommended Architecture:**

1. **Hot Data (Last 7 days):** Redis cache + time-partitioned DB
2. **Warm Data (7-30 days):** Time-partitioned DB with covering indexes
3. **Cold Data (30+ days):** Archived to compressed storage
4. **Celebrities:** Separate fan-out strategy (on-read mixing)
5. **Monitoring:** Track index size, query latency, cache hit rate

**Real-World Examples:**
- **Twitter:** Fan-out on write for regular users, fan-out on read for celebrities
- **Facebook:** Sophisticated caching + mixed fanout strategy
- **Instagram:** Time-partitioned + heavy Redis caching

This demonstrates how indexing strategy must consider the entire system architecture at massive scale.`,
    keyPoints: [
      'Sharding reduces index size but complicates queries',
      'Time-based partitioning keeps hot data fast with small indexes',
      'Fan-out on write trades write complexity for simple, fast reads',
      'Hybrid approaches for different user tiers (regular vs celebrities)',
      'Cache layer essential for massive scale (Redis + time partitions)',
    ],
  },
  {
    id: 'indexing-disc-3',
    question:
      'You have a query that is slow despite having what seems like the right index. Walk through the systematic troubleshooting process you would use to diagnose and fix the issue. What tools and techniques would you use?',
    sampleAnswer: `Systematic approach to diagnosing slow queries:

**Step 1: Gather Information**

\`\`\`sql
-- Get current query
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
WHERE query LIKE '%target_pattern%'
ORDER BY total_time DESC;

-- Check table size
SELECT 
    pg_size_pretty(pg_relation_size('users')) as table_size,
    pg_size_pretty(pg_total_relation_size('users')) as total_size;

-- List existing indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'users';
\`\`\`

**Step 2: Analyze Query Plan**

\`\`\`sql
-- EXPLAIN ANALYZE (actually runs query)
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM users WHERE email = 'john@example.com';
\`\`\`

*Look for red flags:*
- **Seq Scan** instead of Index Scan → index not being used
- **High cost numbers** (>10000) → inefficient plan
- **Large row estimates vs actuals** → outdated statistics
- **Nested Loop with large outer** → JOIN order issue
- **Bitmap Heap Scan** → might need covering index

**Step 3: Common Root Causes**

**Problem 1: Wrong Index**
\`\`\`sql
-- Have: CREATE INDEX idx_name ON users(last_name, first_name);
-- Query: WHERE first_name = 'John'
-- Issue: Can't use index (violates left-prefix rule)

-- Fix: CREATE INDEX idx_first_name ON users(first_name);
\`\`\`

**Problem 2: Type Mismatch**
\`\`\`sql
-- Column: user_id INT
-- Query: WHERE user_id = '123'  -- String literal
-- Issue: Implicit cast prevents index usage

-- Fix: WHERE user_id = 123  -- Use integer literal
\`\`\`

**Problem 3: Function on Indexed Column**
\`\`\`sql
-- Have: CREATE INDEX idx_email ON users(email);
-- Query: WHERE LOWER(email) = 'john@example.com'
-- Issue: Function prevents index usage

-- Fix 1: Expression index
CREATE INDEX idx_email_lower ON users(LOWER(email));

-- Fix 2: Store normalized (better)
ALTER TABLE users ADD COLUMN email_lower VARCHAR(255);
UPDATE users SET email_lower = LOWER(email);
CREATE INDEX idx_email_lower ON users(email_lower);
\`\`\`

**Problem 4: Outdated Statistics**
\`\`\`sql
-- Check when stats were last updated
SELECT 
    schemaname, tablename, last_analyze, last_autoanalyze
FROM pg_stat_user_tables
WHERE tablename = 'users';

-- Force update
ANALYZE users;

-- Re-run EXPLAIN
EXPLAIN ANALYZE SELECT ...;
\`\`\`

**Problem 5: Low Selectivity**
\`\`\`sql
-- Query matches 40% of rows
-- Optimizer chooses Seq Scan over Index Scan

SELECT 
    country,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM users
GROUP BY country;

-- If query matches many rows, index isn't helpful
-- Consider: Partial index for rare values
CREATE INDEX idx_rare_countries ON users(country)
WHERE country NOT IN ('USA', 'UK', 'Canada');
\`\`\`

**Problem 6: Index Bloat**
\`\`\`sql
-- Check index bloat
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'users';

-- Rebuild bloated index
REINDEX INDEX CONCURRENTLY idx_users_email;
\`\`\`

**Step 4: Test Solutions**

\`\`\`sql
-- Create new index
CREATE INDEX CONCURRENTLY idx_test ON users(email, created_at);

-- Force use of specific index
SET enable_seqscan = off;
EXPLAIN ANALYZE SELECT ...;
SET enable_seqscan = on;

-- Compare plans
EXPLAIN (FORMAT JSON) SELECT ...;
\`\`\`

**Step 5: Monitor Impact**

\`\`\`sql
-- Track index usage
SELECT
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Monitor query performance
SELECT 
    query,
    calls,
    mean_time,
    min_time,
    max_time
FROM pg_stat_statements
WHERE query LIKE '%users%'
ORDER BY mean_time DESC;
\`\`\`

**Step 6: Advanced Techniques**

**Technique 1: Covering Index**
\`\`\`sql
-- Original: Index Scan + Table Lookup
CREATE INDEX idx_email ON users(email);

-- Improved: Index-Only Scan
CREATE INDEX idx_email_covering ON users(email) 
INCLUDE (first_name, last_name, created_at);
\`\`\`

**Technique 2: Composite Index Reordering**
\`\`\`sql
-- Test different column orders
CREATE INDEX idx_test1 ON users(country, city, age);
CREATE INDEX idx_test2 ON users(city, country, age);

-- Use EXPLAIN to compare
EXPLAIN SELECT * FROM users WHERE city = 'SF' AND country = 'USA';
\`\`\`

**Technique 3: Partial Index**
\`\`\`sql
-- Full index: 10GB
CREATE INDEX idx_all ON orders(user_id, created_at);

-- Partial index: 500MB (only active orders)
CREATE INDEX idx_active ON orders(user_id, created_at)
WHERE status IN ('pending', 'processing');
\`\`\`

**Tools and Utilities:**

1. **pg_stat_statements:** Query performance tracking
2. **EXPLAIN ANALYZE:** Query plan analysis
3. **pgBadger:** Log analysis and visualization
4. **pg_hero:** Index suggestions and bloat detection
5. **DataDog/NewRelic:** APM for production monitoring

**Checklist:**

✅ Run EXPLAIN ANALYZE
✅ Check if index exists
✅ Verify index is being used
✅ Check statistics are current
✅ Confirm no type mismatches
✅ Verify no functions on indexed columns
✅ Check index selectivity
✅ Monitor index bloat
✅ Test alternative indexes
✅ Measure before/after performance

This systematic approach catches 95% of index-related performance issues.`,
    keyPoints: [
      'Start with EXPLAIN ANALYZE to see actual query plan',
      'Common issues: wrong column order, type mismatch, function on column',
      'Check statistics currency with ANALYZE command',
      'Consider covering indexes, partial indexes, functional indexes',
      'Use pg_stat_statements and monitoring tools to track impact',
    ],
  },
];
