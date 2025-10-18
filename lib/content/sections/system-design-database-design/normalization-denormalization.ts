/**
 * Database Normalization & Denormalization Section
 */

export const normalizationdenormalizationSection = {
  id: 'normalization-denormalization',
  title: 'Database Normalization & Denormalization',
  content: `Normalization and denormalization represent opposing approaches to database schema design. Understanding when to use each is crucial for building systems that balance data integrity, query performance, and maintenance complexity.

## What is Normalization?

**Normalization** is the process of organizing data to reduce redundancy and improve data integrity by dividing tables and establishing relationships.

### Goals of Normalization:
1. **Eliminate Redundancy:** Store each piece of data only once
2. **Ensure Data Integrity:** Prevent update, insertion, and deletion anomalies
3. **Simplify Maintenance:** Changes in one place propagate everywhere
4. **Reduce Storage:** No duplicate data

### Anomalies Without Normalization:

**Unnormalized Table:**
\`\`\`
Orders Table:
+----------+-------------+----------------+--------------+---------------+
| order_id | customer_id | customer_name  | customer_email| product_name  |
+----------+-------------+----------------+--------------+---------------+
| 1        | 101         | Alice Smith    | alice@ex.com | Laptop        |
| 2        | 101         | Alice Smith    | alice@ex.com | Mouse         |
| 3        | 102         | Bob Jones      | bob@ex.com   | Keyboard      |
+----------+-------------+----------------+--------------+---------------+
\`\`\`

**Problems:**
1. **Update Anomaly:** If Alice changes email, must update multiple rows
2. **Insertion Anomaly:** Can't add customer without order
3. **Deletion Anomaly:** Deleting Bob's order loses Bob's info entirely
4. **Storage Waste:** Alice's info duplicated

## Normal Forms

### First Normal Form (1NF)

**Rule:** Each column contains atomic (indivisible) values, and each row is unique.

**Violation Example:**
\`\`\`sql
-- BAD: Multiple values in one column
CREATE TABLE orders (
    order_id INT,
    customer_name VARCHAR(255),
    products VARCHAR(1000)  -- "Laptop, Mouse, Keyboard"
);
\`\`\`

**1NF Compliant:**
\`\`\`sql
CREATE TABLE orders (
    order_id INT,
    customer_name VARCHAR(255),
    product VARCHAR(255)
);

-- Separate rows for each product
INSERT INTO orders VALUES (1, 'Alice', 'Laptop');
INSERT INTO orders VALUES (1, 'Alice', 'Mouse');
\`\`\`

**Better (Proper Relationships):**
\`\`\`sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT
);

CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(255)
);
\`\`\`

### Second Normal Form (2NF)

**Rule:** Must be in 1NF AND all non-key columns must depend on the entire primary key (no partial dependencies).

**Relevant for composite primary keys only.**

**Violation Example:**
\`\`\`sql
-- Composite key: (order_id, product_id)
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    customer_name VARCHAR(255),  -- Depends only on order_id
    product_price DECIMAL(10,2), -- Depends only on product_id
    PRIMARY KEY (order_id, product_id)
);
\`\`\`

**Problem:** \`customer_name\` depends only on \`order_id\`, not the full key. \`product_price\` depends only on \`product_id\`.

**2NF Compliant:**
\`\`\`sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_name VARCHAR(255)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_price DECIMAL(10,2)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
\`\`\`

### Third Normal Form (3NF)

**Rule:** Must be in 2NF AND no transitive dependencies (non-key columns depend only on the primary key, not on other non-key columns).

**Violation Example:**
\`\`\`sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    employee_name VARCHAR(255),
    department_id INT,
    department_name VARCHAR(255),    -- Transitive dependency
    department_location VARCHAR(255) -- Transitive dependency
);
-- department_name depends on department_id, not employee_id
\`\`\`

**Problem:** \`department_name\` depends on \`department_id\`, which is not the primary key.

**3NF Compliant:**
\`\`\`sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    employee_name VARCHAR(255),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(255),
    department_location VARCHAR(255)
);
\`\`\`

**Benefits:**
- Update department name in one place
- No redundancy
- Data integrity maintained

### Boyce-Codd Normal Form (BCNF)

**Rule:** Stricter version of 3NF. For every functional dependency X → Y, X must be a superkey.

**Most tables in 3NF are also in BCNF.** BCNF matters when there are overlapping candidate keys.

**Violation Example:**
\`\`\`sql
CREATE TABLE course_instructors (
    course_id INT,
    instructor_id INT,
    instructor_office VARCHAR(50),
    PRIMARY KEY (course_id, instructor_id)
);

-- Functional dependencies:
-- (course_id, instructor_id) → instructor_office
-- instructor_id → instructor_office  (Violation: instructor_id is not a superkey)
\`\`\`

**BCNF Compliant:**
\`\`\`sql
CREATE TABLE course_instructors (
    course_id INT,
    instructor_id INT,
    PRIMARY KEY (course_id, instructor_id),
    FOREIGN KEY (instructor_id) REFERENCES instructors(instructor_id)
);

CREATE TABLE instructors (
    instructor_id INT PRIMARY KEY,
    instructor_name VARCHAR(255),
    instructor_office VARCHAR(50)
);
\`\`\`

### Fourth Normal Form (4NF) and Beyond

**4NF:** Eliminates multi-valued dependencies
**5NF:** Eliminates join dependencies

**In practice, 3NF/BCNF is sufficient for most applications.** Higher normal forms are rarely needed.

## When to Normalize

### ✅ Normalize For:

1. **Write-Heavy Systems:** Data is frequently updated
2. **OLTP (Transactional Systems):** E-commerce, banking, CRM
3. **Data Integrity Critical:** Financial systems, healthcare
4. **Frequent Updates:** User profiles, inventory management
5. **Complex Relationships:** Many-to-many, hierarchical data

**Example: E-commerce Order System**
\`\`\`sql
-- Normalized (good for transactional workload)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    email VARCHAR(255),
    name VARCHAR(255)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10,2)
);
\`\`\`

**Benefits:**
- Update product price in one place → affects all orders going forward
- Customer email change requires single UPDATE
- No duplicate product data
- Easy to maintain referential integrity

## What is Denormalization?

**Denormalization** is the process of intentionally introducing redundancy to improve read performance by reducing JOINs.

### Goals of Denormalization:
1. **Improve Read Performance:** Fewer JOINs = faster queries
2. **Reduce Query Complexity:** Simpler SQL queries
3. **Optimize for Specific Access Patterns:** Pre-computed results
4. **Enable Horizontal Scaling:** NoSQL databases often require denormalization

### Trade-offs:
- **Storage:** More redundant data
- **Consistency:** Harder to keep data in sync
- **Maintenance:** More complex update logic
- **Write Performance:** Multiple places to update

## When to Denormalize

### ✅ Denormalize For:

1. **Read-Heavy Systems:** 10:1 or 100:1 read:write ratio
2. **OLAP (Analytics):** Reporting, dashboards, data warehouses
3. **Performance Critical:** Sub-10ms query requirements
4. **Simple Access Patterns:** Known query patterns
5. **Microservices:** Each service owns its data
6. **NoSQL Databases:** Document stores, key-value stores

**Example: Analytics Dashboard**
\`\`\`sql
-- Denormalized for fast reporting
CREATE TABLE order_summary (
    order_id INT PRIMARY KEY,
    customer_id INT,
    customer_name VARCHAR(255),      -- Denormalized
    customer_email VARCHAR(255),     -- Denormalized
    order_date TIMESTAMP,
    total_amount DECIMAL(10,2),      -- Pre-computed
    product_count INT,               -- Pre-computed
    products TEXT                    -- Denormalized: "Laptop, Mouse"
);

-- Single query, no JOINs, ultra-fast
SELECT * FROM order_summary WHERE order_date >= '2024-01-01';
\`\`\`

**Benefits:**
- **Single query:** No JOINs across 4 tables
- **Fast:** 5ms vs 100ms with JOINs
- **Simple:** Easy to understand and maintain

**Costs:**
- If customer changes email, must update all order_summary rows
- More storage (duplicated customer info)
- Potential inconsistency if updates fail

## Denormalization Strategies

### 1. Duplicating Columns

**Pattern:** Copy frequently accessed columns to avoid JOINs

\`\`\`sql
-- Normalized
CREATE TABLE comments (
    comment_id INT PRIMARY KEY,
    post_id INT,
    user_id INT,
    content TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- To display comments, need JOIN to get username
SELECT c.content, u.username 
FROM comments c
JOIN users u ON c.user_id = u.user_id
WHERE c.post_id = 123;
\`\`\`

**Denormalized:**
\`\`\`sql
CREATE TABLE comments (
    comment_id INT PRIMARY KEY,
    post_id INT,
    user_id INT,
    username VARCHAR(255),  -- Denormalized: copy of users.username
    content TEXT
);

-- No JOIN needed
SELECT content, username 
FROM comments 
WHERE post_id = 123;
\`\`\`

**Trade-off:** If user changes username, must update all their comments.

### 2. Pre-computing Aggregations

**Pattern:** Store computed values instead of calculating on every query

\`\`\`sql
-- Normalized: compute likes on every query
SELECT post_id, COUNT(*) as like_count
FROM likes
WHERE post_id = 123;
\`\`\`

**Denormalized:**
\`\`\`sql
CREATE TABLE posts (
    post_id INT PRIMARY KEY,
    content TEXT,
    like_count INT DEFAULT 0  -- Denormalized: pre-computed
);

-- Update with trigger or application logic
CREATE TRIGGER update_like_count
AFTER INSERT ON likes
FOR EACH ROW
UPDATE posts SET like_count = like_count + 1 WHERE post_id = NEW.post_id;

-- Fast query
SELECT post_id, like_count FROM posts WHERE post_id = 123;
\`\`\`

**Trade-off:** Must maintain like_count on every INSERT/DELETE to likes table.

### 3. Materialized Views

**Pattern:** Pre-compute and store complex query results

\`\`\`sql
-- Complex join query (slow: 500ms)
SELECT 
    p.product_id,
    p.name,
    c.category_name,
    AVG(r.rating) as avg_rating,
    COUNT(r.review_id) as review_count,
    SUM(oi.quantity) as total_sold
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN reviews r ON p.product_id = r.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name, c.category_name;
\`\`\`

**Materialized View:**
\`\`\`sql
-- PostgreSQL
CREATE MATERIALIZED VIEW product_stats AS
SELECT 
    p.product_id,
    p.name,
    c.category_name,
    AVG(r.rating) as avg_rating,
    COUNT(r.review_id) as review_count,
    SUM(oi.quantity) as total_sold
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN reviews r ON p.product_id = r.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name, c.category_name;

-- Refresh periodically
REFRESH MATERIALIZED VIEW product_stats;

-- Fast query (5ms)
SELECT * FROM product_stats WHERE product_id = 123;
\`\`\`

**Trade-off:** Data may be stale between refreshes. Refresh operation can be expensive.

### 4. Array/JSON Columns

**Pattern:** Store related data as nested structures (common in NoSQL)

\`\`\`sql
-- Normalized
CREATE TABLE blog_posts (
    post_id INT PRIMARY KEY,
    title TEXT
);

CREATE TABLE tags (
    tag_id INT PRIMARY KEY,
    tag_name VARCHAR(50)
);

CREATE TABLE post_tags (
    post_id INT,
    tag_id INT,
    PRIMARY KEY (post_id, tag_id)
);
\`\`\`

**Denormalized (PostgreSQL):**
\`\`\`sql
CREATE TABLE blog_posts (
    post_id INT PRIMARY KEY,
    title TEXT,
    tags TEXT[]  -- Array: ['python', 'database', 'tutorial']
);

-- Query without JOIN
SELECT * FROM blog_posts WHERE 'python' = ANY(tags);
\`\`\`

**Denormalized (MongoDB):**
\`\`\`javascript
{
  _id: ObjectId("..."),
  title: "Database Indexing Guide",
  tags: ["database", "indexing", "postgresql"],
  author: {
    name: "Alice",
    email: "alice@example.com"
  },
  comments: [
    { user: "Bob", text: "Great article!", date: ISODate("...") },
    { user: "Carol", text: "Very helpful", date: ISODate("...") }
  ]
}
\`\`\`

**Trade-off:** Harder to enforce referential integrity, query nested data, and update embedded objects.

### 5. Event Sourcing / CQRS

**Pattern:** Separate read and write models

\`\`\`
Write Model (Normalized):
- Transactional database (PostgreSQL)
- Fully normalized for data integrity
- Handles commands: CreateOrder, UpdateUser

Read Model (Denormalized):
- Optimized for queries (Elasticsearch, Redis, denormalized tables)
- Pre-computed, cached, materialized
- Handles queries: GetOrderSummary, SearchProducts

Sync:
- Events from write model update read model
- Eventual consistency
\`\`\`

**Example:**
\`\`\`
Write Model:
orders table (normalized)
order_items table (normalized)
products table (normalized)

→ Event: OrderCreated

Read Model:
order_summary table (denormalized: all info in one row)
user_orders_cache (Redis: list of order summaries)
product_search_index (Elasticsearch)
\`\`\`

## Hybrid Approach (Recommended)

**Most production systems use a combination:**

### Pattern: Normalize Core Data, Denormalize for Performance

\`\`\`sql
-- Normalized source of truth
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(255)
);

CREATE TABLE posts (
    post_id INT PRIMARY KEY,
    user_id INT,
    content TEXT,
    created_at TIMESTAMP,
    like_count INT DEFAULT 0,  -- Denormalized for performance
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE likes (
    like_id INT PRIMARY KEY,
    post_id INT,
    user_id INT,
    FOREIGN KEY (post_id) REFERENCES posts(post_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Update denormalized like_count
CREATE TRIGGER update_post_likes
AFTER INSERT ON likes
FOR EACH ROW
UPDATE posts SET like_count = like_count + 1 WHERE post_id = NEW.post_id;
\`\`\`

**Benefits:**
- \`users\` and \`likes\` tables: normalized (source of truth)
- \`posts.like_count\`: denormalized for fast queries (SELECT like_count from posts)
- Inconsistency is detectable and fixable (recount from likes table)

## Real-World Examples

### Example 1: Social Media Feed

**Normalized (Write Model):**
\`\`\`sql
CREATE TABLE users (user_id, name, email);
CREATE TABLE posts (post_id, user_id, content, created_at);
CREATE TABLE follows (follower_id, following_id);
CREATE TABLE likes (like_id, post_id, user_id);
CREATE TABLE comments (comment_id, post_id, user_id, content);
\`\`\`

**Denormalized (Read Model - Feed View):**
\`\`\`sql
CREATE TABLE feed_items (
    feed_item_id INT PRIMARY KEY,
    user_id INT,              -- whose feed
    post_id INT,
    author_id INT,
    author_name VARCHAR(255), -- Denormalized
    author_avatar_url TEXT,   -- Denormalized
    content TEXT,             -- Denormalized
    like_count INT,           -- Denormalized
    comment_count INT,        -- Denormalized
    created_at TIMESTAMP
);

-- Ultra-fast feed query
SELECT * FROM feed_items WHERE user_id = 123 ORDER BY created_at DESC LIMIT 50;
\`\`\`

**Strategy:**
- Write: Insert into normalized tables
- Background job: Fan-out posts to followers' feeds (denormalized)
- Read: Query denormalized feed_items (no JOINs, <10ms)

### Example 2: E-commerce Product Catalog

**Normalized (Write Model):**
\`\`\`sql
CREATE TABLE products (product_id, name, price, category_id);
CREATE TABLE categories (category_id, name, parent_category_id);
CREATE TABLE reviews (review_id, product_id, user_id, rating, text);
CREATE TABLE inventory (product_id, warehouse_id, quantity);
\`\`\`

**Denormalized (Search Index - Elasticsearch):**
\`\`\`json
{
  "product_id": 123,
  "name": "Wireless Mouse",
  "price": 29.99,
  "category": "Electronics > Computers > Accessories",  // Denormalized
  "avg_rating": 4.5,        // Denormalized
  "review_count": 342,      // Denormalized
  "total_inventory": 1523,  // Denormalized
  "in_stock": true          // Denormalized
}
\`\`\`

**Strategy:**
- Write: Update normalized PostgreSQL tables
- Sync: Replicate changes to Elasticsearch (denormalized)
- Read: Search from Elasticsearch (full-text, faceting, fast)

### Example 3: Time-Series Analytics

**Normalized (Raw Data):**
\`\`\`sql
CREATE TABLE events (
    event_id BIGSERIAL,
    user_id INT,
    event_type VARCHAR(50),
    created_at TIMESTAMP,
    metadata JSONB
);
-- 1 billion rows
\`\`\`

**Denormalized (Aggregated Data):**
\`\`\`sql
-- Rollup tables for different time granularities
CREATE TABLE events_hourly (
    hour TIMESTAMP,
    event_type VARCHAR(50),
    event_count INT,
    unique_users INT,
    PRIMARY KEY (hour, event_type)
);

CREATE TABLE events_daily (
    date DATE,
    event_type VARCHAR(50),
    event_count INT,
    unique_users INT,
    PRIMARY KEY (date, event_type)
);

-- Background job: aggregate raw events into rollup tables
INSERT INTO events_hourly 
SELECT 
    date_trunc('hour', created_at) as hour,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users
FROM events
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY hour, event_type;
\`\`\`

**Strategy:**
- Write: Append-only to events table (fast inserts)
- Background: Roll up into hourly/daily tables
- Read: Query rollup tables (1000x fewer rows)

## Decision Framework

### Choose Normalization When:
- ✅ Data integrity is critical
- ✅ Frequent updates to shared data
- ✅ Write-heavy workload
- ✅ Complex relationships
- ✅ Source of truth / transactional system

### Choose Denormalization When:
- ✅ Read-heavy workload (10:1+ read:write)
- ✅ Known access patterns
- ✅ Performance critical (sub-10ms queries)
- ✅ Tolerate eventual consistency
- ✅ Analytics / reporting

### Hybrid (Best for Most Systems):
- ✅ Normalize write path (source of truth)
- ✅ Denormalize read path (performance)
- ✅ Use caching, materialized views, search indexes
- ✅ Event-driven sync between models

## Interview Tips

**Q: "When would you denormalize a database?"**
- Read-heavy workloads with known query patterns
- Performance requirements that can't be met with indexes alone
- Analytics/reporting systems where slight staleness is acceptable
- Microservices where each service denormalizes data it needs

**Q: "What are the risks of denormalization?"**
- Data inconsistency if updates fail partially
- More complex update logic (multiple places to update)
- Increased storage requirements
- Harder to maintain referential integrity

**Q: "How do you maintain consistency in denormalized systems?"**
- Database triggers (simple cases)
- Application-level logic with transactions
- Event-driven architecture (eventual consistency)
- Background jobs to reconcile/repair inconsistencies
- Monitoring and alerting for drift

**Q: "Design a normalized schema for X" or "Why would you denormalize Y?"**
- Start with normalized design (show you understand normalization)
- Identify query patterns and bottlenecks
- Propose specific denormalization strategies
- Discuss trade-offs and mitigation strategies

## Key Takeaways

1. **Normalization eliminates redundancy and ensures data integrity**
2. **Denormalization trades consistency for read performance**
3. **3NF is sufficient for most transactional systems**
4. **Most production systems use hybrid approaches**
5. **Normalize the write path (source of truth)**
6. **Denormalize the read path (performance)**
7. **Use materialized views, caching, and search indexes for denormalization**
8. **Event-driven architectures enable normalized writes + denormalized reads**
9. **Choose based on workload: write-heavy → normalize, read-heavy → denormalize**
10. **Consistency mechanisms are essential for denormalized systems**

## Summary

Normalization (3NF) is ideal for write-heavy transactional systems where data integrity is critical. Denormalization is ideal for read-heavy systems where performance matters more than strict consistency. Most production systems use a hybrid approach: normalize the source of truth, denormalize for specific access patterns using materialized views, caching, or separate read models (CQRS). The key is understanding your workload characteristics and making intentional trade-offs.
`,
};
