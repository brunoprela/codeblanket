/**
 * Normalization vs Denormalization Section
 */

export const normalizationvsdenormalizationSection = {
  id: 'normalization-vs-denormalization',
  title: 'Normalization vs Denormalization',
  content: `Database normalization and denormalization represent a fundamental trade-off between data integrity and query performance. Understanding when to apply each is crucial for system design.

## Definitions

**Normalization**: Organizing database schema to **reduce redundancy** and improve data integrity.
- Store each piece of data only once
- Use foreign keys to link related data
- Follow normal forms (1NF, 2NF, 3NF, BCNF)

**Denormalization**: Intentionally adding **redundancy** to database schema to improve read performance.
- Duplicate data across tables
- Pre-compute aggregations
- Trade storage and write complexity for fast reads

---

## Database Normalization

### Normal Forms

**1NF (First Normal Form)**:
- Each column contains atomic values (no lists/arrays)
- Each row is unique (has primary key)

**2NF (Second Normal Form)**:
- Must be in 1NF
- No partial dependencies (non-key columns depend on entire primary key)

**3NF (Third Normal Form)**:
- Must be in 2NF
- No transitive dependencies (non-key columns don't depend on other non-key columns)

**BCNF (Boyce-Codd Normal Form)**:
- Stricter version of 3NF
- Every determinant is a candidate key

### Example: E-commerce Database (Normalized)

**Unnormalized** (has redundancy):
\`\`\`
Orders table:
order_id | user_id | user_name | user_email | product_id | product_name | product_price | quantity
1        | 101     | Alice     | a@email.com| 501        | Laptop       | $1000         | 1
2        | 101     | Alice     | a@email.com| 502        | Mouse        | $20           | 2
3        | 102     | Bob       | b@email.com| 501        | Laptop       | $1000         | 1
\`\`\`

**Problems**:
- User data (name, email) repeated for every order
- Product data (name, price) repeated for every order
- If Alice changes email, must update multiple rows
- If Laptop price changes, must update multiple rows

**Normalized** (3NF):
\`\`\`
Users table:
user_id | name  | email
101     | Alice | a@email.com
102     | Bob   | b@email.com

Products table:
product_id | name   | price
501        | Laptop | $1000
502        | Mouse  | $20

Orders table:
order_id | user_id | created_at
1        | 101     | 2024-01-01
2        | 101     | 2024-01-02
3        | 102     | 2024-01-03

Order_Items table:
order_item_id | order_id | product_id | quantity | price_at_purchase
1             | 1        | 501        | 1        | $1000
2             | 2        | 502        | 2        | $20
3             | 3        | 501        | 1        | $1000
\`\`\`

**Benefits**:
- Each piece of data stored once
- Update user email in one place
- No anomalies (insert, update, delete)
- Data integrity maintained

**Cost**:
- Need JOINs to get complete order data
- Slower queries (multiple table access)

---

## Database Denormalization

### When to Denormalize

Denormalize when **read performance** is more important than:
- Storage efficiency
- Write performance
- Update complexity
- Data integrity

### Denormalization Techniques

**1. Duplicate Data**

Add redundant columns to avoid JOINs.

**Example**: Add user_name to orders table
\`\`\`
Orders table:
order_id | user_id | user_name | created_at
1        | 101     | Alice     | 2024-01-01
\`\`\`

**Benefit**: No JOIN with users table to get user name
**Cost**: Must update user_name in multiple places when user changes name

---

**2. Pre-compute Aggregations**

Store computed values instead of calculating on each query.

**Example**: Store order total in orders table
\`\`\`
Orders table:
order_id | user_id | created_at  | total_amount
1        | 101     | 2024-01-01  | $1000
\`\`\`

**Without denormalization**:
\`\`\`sql
SELECT order_id, SUM(quantity * price_at_purchase) as total
FROM order_items
GROUP BY order_id
\`\`\`

**With denormalization**:
\`\`\`sql
SELECT order_id, total_amount FROM orders
\`\`\`

**Benefit**: Instant query (no aggregation)
**Cost**: Must update total_amount when order items change

---

**3. Materialized Views**

Pre-compute complex queries and store results.

**Example**: Product revenue report
\`\`\`
Product_Revenue (materialized view):
product_id | product_name | total_revenue | total_quantity_sold
501        | Laptop       | $50,000       | 50
502        | Mouse        | $2,000        | 100
\`\`\`

**Benefit**: Complex aggregation pre-computed
**Cost**: Need to refresh view periodically

---

## Real-World Examples

### Example 1: Twitter Timeline

**Normalized approach** (Pull model):
\`\`\`
When user views timeline:
1. SELECT user_ids FROM followers WHERE follower_id = current_user
2. SELECT * FROM tweets WHERE user_id IN (user_ids) ORDER BY created_at DESC LIMIT 50
\`\`\`

**Problem**: Slow (need to query all followed users' tweets on every timeline view)

**Denormalized approach** (Push model):
\`\`\`
When user tweets:
1. INSERT INTO tweets (content, user_id, created_at)
2. INSERT INTO timelines (follower_id, tweet_id, created_at) for each follower

When user views timeline:
1. SELECT tweet_id FROM timelines WHERE follower_id = current_user ORDER BY created_at DESC LIMIT 50
2. SELECT * FROM tweets WHERE tweet_id IN (tweet_ids)
\`\`\`

**Benefit**: Fast timeline loads (data pre-computed)
**Cost**: High write complexity (one tweet → millions of timeline inserts for celebrity)

**Twitter\'s hybrid approach**:
- Denormalized (push) for regular users (<1M followers)
- Normalized (pull) for celebrities (>1M followers)
- Merge both on timeline load

---

### Example 2: E-commerce Product Listings

**Normalized**:
\`\`\`
Products table: product_id, name, category_id
Categories table: category_id, name
Reviews table: review_id, product_id, rating, comment
\`\`\`

**Query to display product listing**:
\`\`\`sql
SELECT p.*, c.name as category_name, AVG(r.rating) as avg_rating, COUNT(r.review_id) as review_count
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id
\`\`\`

**Problem**: Complex JOIN + aggregation on every product listing page

**Denormalized**:
\`\`\`
Products table:
product_id | name | category_id | category_name | avg_rating | review_count
501        | Laptop | 1         | Electronics   | 4.5        | 250
\`\`\`

**Query**:
\`\`\`sql
SELECT * FROM products WHERE category_id = 1
\`\`\`

**Benefit**: Simple, fast query (no JOINs, no aggregations)
**Cost**: When new review added, must update products.avg_rating and products.review_count

---

## NoSQL Denormalization

NoSQL databases often **require** denormalization because they don't support JOINs.

### Example: MongoDB User Profile

**Normalized approach** (anti-pattern in NoSQL):
\`\`\`javascript
// Users collection
{ _id: 101, name: "Alice", email: "a@email.com" }

// Posts collection
{ _id: 1, user_id: 101, content: "Hello world" }

// To display post with user name, need application-level JOIN (slow)
\`\`\`

**Denormalized approach** (recommended in NoSQL):
\`\`\`javascript
// Posts collection (embeds user data)
{
  _id: 1,
  content: "Hello world",
  author: {
    id: 101,
    name: "Alice",
    avatar_url: "https://..."
  },
  created_at: "2024-01-01"
}
\`\`\`

**Benefit**: Single query to get post + author data
**Cost**: If Alice changes name, must update all her posts

**When to denormalize in NoSQL**:
- Data read together frequently
- Related data changes infrequently
- Read:write ratio is high (10:1 or higher)

---

## Trade-off Analysis

### Normalization Advantages

✅ **Data integrity**: Single source of truth, no duplicates
✅ **Easier updates**: Change data in one place
✅ **Less storage**: No redundancy
✅ **Enforced consistency**: Foreign keys prevent orphaned records

### Normalization Disadvantages

❌ **Slower reads**: Need JOINs to get complete data
❌ **Complex queries**: Multiple tables, harder to write and optimize
❌ **Poor scalability**: JOINs don't scale horizontally (difficult to shard)

---

### Denormalization Advantages

✅ **Fast reads**: Data pre-computed and co-located
✅ **Simple queries**: No JOINs needed
✅ **Better scalability**: Easier to shard (no JOINs across shards)
✅ **Lower latency**: Single table access

### Denormalization Disadvantages

❌ **Data redundancy**: More storage required
❌ **Complex writes**: Must update multiple places
❌ **Data inconsistency risk**: If updates fail partially
❌ **Stale data**: Aggregations may not be real-time

---

## Decision Framework

### Use Normalization When:
- **Write-heavy workload**: Lots of inserts, updates, deletes
- **Data integrity critical**: Financial data, user accounts
- **Storage expensive**: Redundancy costs too much
- **Small scale**: JOINs perform acceptably (<1M rows)
- **OLTP systems**: Transactional databases

### Use Denormalization When:
- **Read-heavy workload**: 10:1 or higher read:write ratio
- **Performance critical**: Low latency required (<100ms)
- **Large scale**: Millions of queries per second
- **NoSQL database**: JOINs not supported or expensive
- **Analytics systems**: OLAP, data warehouses

---

## Hybrid Approach

Most production systems use **both** normalization and denormalization.

### Pattern: Normalized Write, Denormalized Read

**Write path** (normalized):
\`\`\`
Write to normalized tables (users, products, orders, order_items)
Maintain data integrity with foreign keys
Transactional updates
\`\`\`

**Read path** (denormalized):
\`\`\`
Background job aggregates data into denormalized views
Materialized views refreshed periodically
Read replicas with denormalized schema
\`\`\`

**Example**: E-commerce
- Orders written to normalized tables (immediate)
- Analytics dashboard reads from denormalized tables (refreshed every 5 minutes)

---

## Common Mistakes

### ❌ Mistake 1: Premature Denormalization

**Problem**: Denormalizing before you know query patterns

**Better**: Start normalized, denormalize specific tables based on actual performance data

### ❌ Mistake 2: Full Denormalization in SQL

**Problem**: Making SQL database look like NoSQL (one giant table)

**Better**: Use SQL's strengths (JOINs, transactions) and denormalize selectively

### ❌ Mistake 3: Ignoring Consistency

**Problem**: Denormalizing without plan to keep duplicated data consistent

**Better**: Implement triggers, application-level consistency checks, or accept eventual consistency

### ❌ Mistake 4: Not Using Materialized Views

**Problem**: Manually managing denormalized aggregations

**Better**: Use database's materialized views feature (PostgreSQL, Oracle)

---

## Best Practices

### ✅ 1. Start Normalized

Begin with normalized schema. Profile queries. Denormalize only problem areas.

### ✅ 2. Document Denormalization

Comment why denormalization was needed. Future developers need to understand trade-offs.

### ✅ 3. Maintain Consistency

Use database triggers or application code to keep denormalized data consistent.

**Example**: PostgreSQL trigger
\`\`\`sql
CREATE TRIGGER update_order_total
AFTER INSERT OR UPDATE OR DELETE ON order_items
FOR EACH ROW
EXECUTE FUNCTION update_order_total_function();
\`\`\`

### ✅ 4. Monitor Staleness

If using eventual consistency, monitor how stale data can get. Alert if exceeds threshold.

### ✅ 5. Use Caching Before Denormalization

Try caching query results (Redis) before denormalizing schema. Caching is easier to change.

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd recommend **[normalized/denormalized/hybrid]** approach:

**Normalized for**:
- [Write-heavy tables like user accounts, transactions]
- Benefit: Data integrity, easier updates
- Trade-off: Slower reads due to JOINs

**Denormalized for**:
- [Read-heavy tables like product listings, timelines]
- Benefit: Fast queries (no JOINs), better scalability
- Trade-off: More storage, complex writes

**Consistency strategy**:
- [How to keep denormalized data consistent: triggers, async jobs, eventual consistency]

**Example**: Amazon normalizes order tables (integrity) but denormalizes product catalog (speed)."

---

## Summary Table

| Aspect | Normalization | Denormalization |
|--------|--------------|-----------------|
| **Goal** | Data integrity | Query performance |
| **Redundancy** | Minimal | Intentional duplication |
| **Storage** | Less | More |
| **Write Performance** | Faster (single write) | Slower (multiple writes) |
| **Read Performance** | Slower (JOINs) | Faster (pre-computed) |
| **Consistency** | Strong (single source) | Eventual (duplicates) |
| **Scalability** | Harder (JOINs) | Easier (no JOINs) |
| **Use Case** | OLTP, write-heavy | OLAP, read-heavy |
| **Example** | User accounts | Twitter timeline |

---

## Key Takeaways

✅ Normalization reduces redundancy, improves integrity, but slows reads (JOINs)
✅ Denormalization duplicates data for fast reads, but complicates writes
✅ Use normalization for write-heavy, critical data (transactions, accounts)
✅ Use denormalization for read-heavy, performance-critical data (timelines, catalogs)
✅ Most systems use hybrid: normalized writes, denormalized reads
✅ NoSQL often requires denormalization (no JOIN support)
✅ Start normalized, denormalize based on actual performance data
✅ Maintain consistency: triggers, async jobs, or accept eventual consistency`,
};
