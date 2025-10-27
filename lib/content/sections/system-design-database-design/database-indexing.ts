/**
 * Database Indexing Section
 */

export const databaseindexingSection = {
  id: 'database-indexing',
  title: 'Database Indexing',
  content: `Database indexing is one of the most powerful tools for query optimization. Understanding how indexes work, their types, trade-offs, and when to use them is essential for building performant systems.

## What is a Database Index?

An index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional storage space and write overhead.

### Without an Index
\`\`\`sql
-- Full table scan: O(n)
SELECT * FROM users WHERE email = 'john@example.com';
-- Database must scan every row to find matching records
\`\`\`

### With an Index
\`\`\`sql
-- Create index
CREATE INDEX idx_users_email ON users (email);

-- Now this query uses the index: O(log n)
SELECT * FROM users WHERE email = 'john@example.com';
-- Database uses B-tree to quickly locate matching rows
\`\`\`

**Performance Impact:**
- 1M rows without index: ~1000ms (full scan)
- 1M rows with index: ~1ms (B-tree lookup)

## Index Data Structures

### 1. B-Tree Indexes (Most Common)

**Structure:**
- Self-balancing tree with nodes containing keys and pointers
- All leaf nodes at the same level
- Each node has multiple children (high branching factor)
- Sorted order maintained

**Characteristics:**
- **Time Complexity:** O(log n) for search, insert, delete
- **Range Queries:** Excellent (sorted order)
- **Memory Efficient:** High branching factor = shallow trees
- **Disk-Friendly:** Optimized for block-based storage

**Use Cases:**
- Primary keys and foreign keys
- Equality and range queries
- ORDER BY and GROUP BY operations
- Most general-purpose indexing needs

**Example:**
\`\`\`sql
-- B-tree automatically used for primary keys
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- B-tree index
    email VARCHAR(255),
    created_at TIMESTAMP
);

-- Explicit B-tree indexes
CREATE INDEX idx_email ON users (email);
CREATE INDEX idx_created_at ON users (created_at);

-- Efficient queries
SELECT * FROM users WHERE email = 'john@example.com';
SELECT * FROM users WHERE created_at >= '2024-01-01';
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;
\`\`\`

### 2. Hash Indexes

**Structure:**
- Hash function maps keys to bucket locations
- Direct O(1) access for exact matches

**Characteristics:**
- **Time Complexity:** O(1) for exact match
- **Range Queries:** Not supported
- **Memory Usage:** Can be memory-intensive
- **Collisions:** Require collision handling

**Use Cases:**
- Exact match queries only
- Cache lookups
- Session management

**Example:**
\`\`\`sql
-- PostgreSQL hash index
CREATE INDEX idx_session_hash ON sessions USING HASH(session_id);

-- Efficient for exact matches only
SELECT * FROM sessions WHERE session_id = 'abc123';

-- NOT efficient (can't use hash index)
SELECT * FROM sessions WHERE session_id > 'abc123';
\`\`\`

**Limitations:**
- No support for: <, >, <=, >=, BETWEEN, LIKE
- Can't be used for sorting
- Not crash-safe in some databases (PostgreSQL < 10)

### 3. Full-Text Indexes (Inverted Indexes)

**Structure:**
- Maps each word/token to documents containing it
- Supports linguistic features (stemming, stop words)

**Characteristics:**
- Optimized for text search
- Supports relevance ranking
- Language-aware tokenization
- Complex query operators (AND, OR, phrase, proximity)

**Use Cases:**
- Search engines
- Document databases
- Content management systems
- E-commerce product search

**Example (PostgreSQL):**
\`\`\`sql
-- Create full-text index
CREATE INDEX idx_articles_fulltext
ON articles USING GIN(to_tsvector('english', title || ' ' || content));

-- Full-text search queries
SELECT * FROM articles
WHERE to_tsvector('english', title || ' ' || content)
@@ to_tsquery('english', 'database & indexing');

-- With ranking
SELECT *, ts_rank (to_tsvector('english', content), query) AS rank
FROM articles, to_tsquery('english', 'database & indexing') query
WHERE to_tsvector('english', content) @@ query
ORDER BY rank DESC;
\`\`\`

**Example (Elasticsearch):**
\`\`\`json
{
  "mappings": {
    "properties": {
      "title": { "type": "text", "analyzer": "english" },
      "content": { "type": "text", "analyzer": "english" }
    }
  }
}
\`\`\`

### 4. Geospatial Indexes (R-Tree, Quadtree)

**Structure:**
- Hierarchical spatial partitioning
- Bounding boxes for efficient spatial queries

**Use Cases:**
- Location-based services
- GIS applications
- Proximity searches

**Example (PostgreSQL with PostGIS):**
\`\`\`sql
-- Create spatial index
CREATE INDEX idx_locations_geom ON locations USING GIST(geom);

-- Find nearby locations
SELECT * FROM locations
WHERE ST_DWithin(
    geom,
    ST_MakePoint(-122.4194, 37.7749)::geography,
    1000  -- 1km radius
);
\`\`\`

**Example (MongoDB):**
\`\`\`javascript
// Create 2dsphere index
db.restaurants.createIndex({ location: "2dsphere" });

// Find restaurants within 1km
db.restaurants.find({
  location: {
    $near: {
      $geometry: { type: "Point", coordinates: [-122.4194, 37.7749] },
      $maxDistance: 1000
    }
  }
});
\`\`\`

## Types of Indexes by Column Count

### 1. Single-Column Index

**Definition:** Index on one column only

\`\`\`sql
CREATE INDEX idx_email ON users (email);
\`\`\`

**Use Cases:**
- Primary keys
- Foreign keys
- Frequently queried single columns

### 2. Composite (Multi-Column) Index

**Definition:** Index on multiple columns

\`\`\`sql
CREATE INDEX idx_user_location ON users (country, city, zip_code);
\`\`\`

**Column Order Matters:**
\`\`\`sql
-- This index can efficiently handle:
WHERE country = 'USA'                           -- ✅ Uses index
WHERE country = 'USA' AND city = 'San Francisco' -- ✅ Uses index
WHERE country = 'USA' AND city = 'SF' AND zip_code = '94102' -- ✅ Uses index

-- This index CANNOT efficiently handle:
WHERE city = 'San Francisco'                    -- ❌ Can't use index
WHERE zip_code = '94102'                        -- ❌ Can't use index
WHERE city = 'SF' AND zip_code = '94102'       -- ❌ Can't use index
\`\`\`

**Left-Prefix Rule:**
- Index can be used if query filters match leftmost columns
- Think of it like a phone book: sorted by last name, then first name
- You can find "Smith, John" but not just "John"

**Best Practices:**1. **Order by Selectivity:** Most selective columns first
2. **Order by Query Patterns:** Match common WHERE clauses
3. **Consider Equality vs Range:** Equality filters first, range filters last

\`\`\`sql
-- Good: Equality first, range last
CREATE INDEX idx_orders ON orders (user_id, status, created_at);

-- Supports these efficiently:
WHERE user_id = 123 AND status = 'active' AND created_at > '2024-01-01'
WHERE user_id = 123 AND status = 'active'
WHERE user_id = 123
\`\`\`

### 3. Covering Index (Include Columns)

**Definition:** Index contains all columns needed by query

\`\`\`sql
-- PostgreSQL
CREATE INDEX idx_users_email_covering
ON users (email) INCLUDE (first_name, last_name);

-- This query can be satisfied entirely from the index
SELECT first_name, last_name FROM users WHERE email = 'john@example.com';
-- No need to access table rows (index-only scan)
\`\`\`

**Benefits:**
- Avoid table lookups (faster)
- Reduce I/O operations
- Better for frequently accessed columns

**Trade-offs:**
- Larger index size
- Slower writes
- More storage

## Partial Indexes (Filtered Indexes)

**Definition:** Index only a subset of rows

\`\`\`sql
-- Index only active users
CREATE INDEX idx_active_users ON users (email) WHERE status = 'active';

-- Efficient for queries matching the filter
SELECT * FROM users WHERE email = 'john@example.com' AND status = 'active';
\`\`\`

**Benefits:**
- Smaller index size
- Faster writes (fewer rows indexed)
- Faster queries (smaller index to search)

**Use Cases:**
- Soft deletes: \`WHERE deleted_at IS NULL\`
- Status filters: \`WHERE status = 'active'\`
- Recent data: \`WHERE created_at > NOW() - INTERVAL '30 days'\`

\`\`\`sql
-- E-commerce example
CREATE INDEX idx_pending_orders ON orders (user_id, created_at)
WHERE status = 'pending';

-- Much smaller than indexing all orders
-- Perfect for "My Pending Orders" page
\`\`\`

## Unique Indexes and Constraints

\`\`\`sql
-- Ensure email uniqueness
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- Equivalent to:
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE(email);
\`\`\`

**Composite Unique Indexes:**
\`\`\`sql
-- Unique combination of user_id and product_id
CREATE UNIQUE INDEX idx_cart_items_unique
ON cart_items (user_id, product_id);

-- Allows: user_id=1, product_id=100 and user_id=2, product_id=100
-- Prevents: duplicate (user_id=1, product_id=100)
\`\`\`

## Expression Indexes (Functional Indexes)

**Definition:** Index on a function/expression result

\`\`\`sql
-- Case-insensitive email lookup
CREATE INDEX idx_email_lower ON users(LOWER(email));

SELECT * FROM users WHERE LOWER(email) = 'john@example.com';
\`\`\`

\`\`\`sql
-- Extract year from timestamp
CREATE INDEX idx_orders_year ON orders(EXTRACT(YEAR FROM created_at));

SELECT * FROM orders WHERE EXTRACT(YEAR FROM created_at) = 2024;
\`\`\`

\`\`\`sql
-- JSON field indexing
CREATE INDEX idx_metadata_type ON events((metadata->>'event_type'));

SELECT * FROM events WHERE metadata->>'event_type' = 'click';
\`\`\`

## Index Trade-offs

### Storage Overhead

**Primary Table:**
\`\`\`
users table: 1GB
\`\`\`

**Indexes:**
\`\`\`
idx_email: 200MB
idx_created_at: 150MB
idx_country_city: 300MB
idx_name_fulltext: 500MB
Total: 1.15GB of additional storage
\`\`\`

**Rule of Thumb:** Indexes can easily double your storage requirements

### Write Performance Impact

**Without Indexes:**
\`\`\`
INSERT: 1ms
UPDATE: 1ms
DELETE: 1ms
\`\`\`

**With 5 Indexes:**
\`\`\`
INSERT: 3-5ms  (must update all indexes)
UPDATE: 3-5ms  (if indexed columns change)
DELETE: 3-5ms  (must update all indexes)
\`\`\`

**Write Amplification:**
- Each write operation requires updating multiple indexes
- More indexes = slower writes
- B-tree rebalancing adds overhead

### Read Performance Improvement

**Dramatic improvements:**
\`\`\`
Without index: SELECT in 1000ms (full table scan)
With index:    SELECT in 1ms    (index seek)
\`\`\`

**When Index Isn't Used:**
- Query doesn't match index columns
- Data type mismatch
- Function applied to indexed column (without expression index)
- Low selectivity (too many matching rows)
- Database optimizer chooses full scan (small tables)

## When to Create Indexes

### ✅ Create Indexes For:

1. **Primary Keys:** Usually automatic
2. **Foreign Keys:** Essential for JOIN performance
3. **WHERE Clauses:** Frequently filtered columns
4. **ORDER BY Columns:** For sorting
5. **GROUP BY Columns:** For aggregations
6. **JOIN Columns:** Both sides of join
7. **Unique Constraints:** Data integrity + performance
8. **Covering Common Queries:** Include frequently accessed columns

### ❌ Don't Create Indexes For:

1. **Small Tables:** Full scan is faster (< 1000 rows)
2. **Low Selectivity:** Column with few distinct values
3. **Frequently Updated Columns:** Write overhead > read benefit
4. **Rarely Queried Columns:** No benefit
5. **Write-Heavy Tables:** Indexes slow down writes significantly

## Index Maintenance

### Index Fragmentation

Over time, indexes become fragmented due to insertions/deletions:

\`\`\`sql
-- PostgreSQL: Check index bloat
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty (pg_relation_size (indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size (indexrelid) DESC;

-- Rebuild fragmented index
REINDEX INDEX idx_users_email;
REINDEX TABLE users;  -- Rebuild all indexes on table
\`\`\`

\`\`\`sql
-- MySQL: Optimize table (rebuilds indexes)
OPTIMIZE TABLE users;

-- Check index cardinality
SHOW INDEX FROM users;
\`\`\`

### Index Statistics

Databases use statistics to decide whether to use an index:

\`\`\`sql
-- PostgreSQL: Update statistics
ANALYZE users;
ANALYZE;  -- All tables

-- MySQL
ANALYZE TABLE users;
\`\`\`

### Monitoring Index Usage

\`\`\`sql
-- PostgreSQL: Find unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty (pg_relation_size (indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size (indexrelid) DESC;

-- Drop unused indexes to improve write performance
DROP INDEX idx_rarely_used;
\`\`\`

## Real-World Examples

### E-commerce Product Search

\`\`\`sql
-- Product table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    stock_quantity INTEGER,
    created_at TIMESTAMP,
    search_vector TSVECTOR
);

-- Indexes for different query patterns
CREATE INDEX idx_products_category ON products (category, price);
CREATE INDEX idx_products_price ON products (price) WHERE stock_quantity > 0;
CREATE INDEX idx_products_search ON products USING GIN(search_vector);
CREATE INDEX idx_products_recent ON products (created_at DESC)
    WHERE created_at > NOW() - INTERVAL '30 days';

-- Query patterns these support:
-- 1. Browse by category sorted by price
SELECT * FROM products
WHERE category = 'electronics'
ORDER BY price;

-- 2. Filter available products by price range
SELECT * FROM products
WHERE price BETWEEN 100 AND 500
AND stock_quantity > 0;

-- 3. Full-text search
SELECT * FROM products
WHERE search_vector @@ to_tsquery('laptop');

-- 4. Recently added products
SELECT * FROM products
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
\`\`\`

### Social Media Feed

\`\`\`sql
-- Posts table
CREATE TABLE posts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    content TEXT,
    created_at TIMESTAMP,
    visibility VARCHAR(20)
);

-- Composite index for user's feed
CREATE INDEX idx_posts_user_timeline
ON posts (user_id, created_at DESC)
WHERE visibility = 'public';

-- Covering index for timeline queries
CREATE INDEX idx_posts_feed_covering
ON posts (user_id, created_at DESC)
INCLUDE (content, visibility);

-- Efficient feed query
SELECT content, created_at
FROM posts
WHERE user_id IN (1, 2, 3, 4, 5)  -- Following list
AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC
LIMIT 50;
\`\`\`

### Analytics Dashboard

\`\`\`sql
-- Events table (time-series data)
CREATE TABLE events (
    id BIGSERIAL,
    event_type VARCHAR(50),
    user_id BIGINT,
    created_at TIMESTAMP,
    metadata JSONB
);

-- Partition by time (separate indexes per partition)
CREATE TABLE events_2024_01 PARTITION OF events
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Indexes optimized for analytics queries
CREATE INDEX idx_events_type_time ON events_2024_01(event_type, created_at);
CREATE INDEX idx_events_user ON events_2024_01(user_id, created_at);
CREATE INDEX idx_events_metadata ON events_2024_01 USING GIN(metadata);

-- Fast aggregation queries
SELECT event_type, COUNT(*)
FROM events
WHERE created_at >= '2024-01-01'
AND created_at < '2024-02-01'
GROUP BY event_type;
\`\`\`

## Database-Specific Features

### PostgreSQL

\`\`\`sql
-- BRIN indexes (Block Range INdex) for large sequential data
CREATE INDEX idx_logs_timestamp ON logs USING BRIN(timestamp);
-- Much smaller than B-tree, good for time-series

-- Concurrent index creation (no table lock)
CREATE INDEX CONCURRENTLY idx_users_email ON users (email);
\`\`\`

### MySQL

\`\`\`sql
-- InnoDB automatically includes primary key in secondary indexes
CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(255)
);
CREATE INDEX idx_email ON users (email);
-- Secondary index stores: (email, id)
-- Allows index-only scans for: SELECT id FROM users WHERE email = '...'
\`\`\`

### MongoDB

\`\`\`javascript
// Compound index
db.products.createIndex({ category: 1, price: -1 });

// TTL index (auto-expire documents)
db.sessions.createIndex({ createdAt: 1 }, { expireAfterSeconds: 3600 });

// Partial index
db.users.createIndex(
  { email: 1 },
  { unique: true, partialFilterExpression: { status: "active" } }
);
\`\`\`

## Interview Tips

### Common Questions

**Q: "How would you optimize this slow query?"**1. Check if there's a WHERE clause → index those columns
2. Check for JOIN → index foreign keys
3. Check ORDER BY/GROUP BY → consider composite index
4. Use EXPLAIN to verify

**Q: "Why might an index not be used?"**
- Wrong column order in composite index
- Function applied to column without expression index
- Type mismatch (string vs integer)
- Low selectivity (optimizer chooses full scan)
- Statistics outdated

**Q: "What are the downsides of indexes?"**
- Storage overhead (can double database size)
- Write performance impact (3-5x slower with many indexes)
- Maintenance overhead (fragmentation, statistics)
- Diminishing returns (too many indexes)

**Q: "How do you decide which indexes to create?"**1. Analyze query patterns (WHERE, JOIN, ORDER BY)
2. Identify slow queries (query logs, APM tools)
3. Use EXPLAIN to verify query plans
4. Monitor index usage (remove unused indexes)
5. Balance read vs write workload

### Design Patterns

**Pattern 1: Composite Index for Common Query**
\`\`\`sql
-- Query: Get user's recent orders
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at DESC);
\`\`\`

**Pattern 2: Covering Index for Performance**
\`\`\`sql
-- Query needs: user_id, status, total
CREATE INDEX idx_orders_covering
ON orders (user_id, created_at)
INCLUDE (status, total);
\`\`\`

**Pattern 3: Partial Index for Selective Data**
\`\`\`sql
-- Only index pending/processing orders (5% of data)
CREATE INDEX idx_active_orders
ON orders (user_id, created_at)
WHERE status IN ('pending', 'processing');
\`\`\`

## Key Takeaways

1. **Indexes trade storage and write performance for read performance**2. **B-tree indexes are the default and work for most use cases**3. **Composite index column order matters (left-prefix rule)**4. **Create indexes on WHERE, JOIN, ORDER BY, and GROUP BY columns**5. **Don't over-index: each index slows down writes**6. **Use EXPLAIN to verify query plans and index usage**7. **Monitor and maintain indexes (rebuild fragmented, drop unused)**8. **Consider partial indexes for frequently queried subsets**9. **Covering indexes eliminate table lookups but increase size**10. **Index selectivity matters: high selectivity = better performance**

## Summary

Database indexing is a fundamental optimization technique. B-tree indexes provide O(log n) lookups vs O(n) full scans. Composite indexes support multiple columns but must match query patterns (left-prefix rule). Specialized indexes (hash, full-text, spatial) optimize specific use cases. Every index trades storage and write performance for faster reads, so careful analysis of query patterns and workload characteristics is essential.
`,
};
