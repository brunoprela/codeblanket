/**
 * SQL vs NoSQL Section
 */

export const sqlvsnosqlSection = {
  id: 'sql-vs-nosql',
  title: 'SQL vs NoSQL',
  content: `Choosing between SQL and NoSQL databases is one of the most fundamental architectural decisions. Each has distinct strengths, and understanding when to use each is critical for system design.

## Definitions

**SQL (Relational) Databases**:
- Store data in **tables** with predefined schema
- Use **SQL** (Structured Query Language) for queries
- ACID transactions (Atomicity, Consistency, Isolation, Durability)
- Examples: PostgreSQL, MySQL, Oracle, SQL Server

**NoSQL (Non-Relational) Databases**:
- Store data in flexible formats (documents, key-value, graphs, columns)
- Schema-less or schema-flexible
- BASE properties (Basically Available, Soft state, Eventually consistent)
- Examples: MongoDB (document), Redis (key-value), Cassandra (column-family), Neo4j (graph)

---

## SQL Databases in Detail

### Structure

**Schema**: Fixed, defined upfront
\`\`\`sql
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
\`\`\`

**Relationships**: Foreign keys link tables
\`\`\`sql
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT REFERENCES users (id),
  total DECIMAL(10,2),
  created_at TIMESTAMP
);
\`\`\`

### ACID Properties

**Atomicity**: Transaction all-or-nothing
\`\`\`sql
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT; -- Both succeed or both rollback
\`\`\`

**Consistency**: Database always in valid state (constraints enforced)

**Isolation**: Concurrent transactions don't interfere

**Durability**: Committed data persists (even if system crashes)

### Advantages

✅ **Strong consistency**: Always see latest data
✅ **ACID transactions**: Perfect for financial data
✅ **Complex queries**: JOINs, aggregations, subqueries
✅ **Data integrity**: Foreign keys, constraints prevent bad data
✅ **Mature ecosystem**: 40+ years of development, well-understood
✅ **Standardized**: SQL is universal across databases

### Disadvantages

❌ **Schema rigidity**: Changing schema difficult at scale
❌ **Vertical scaling**: Hard to horizontally scale (sharding complex)
❌ **Fixed data model**: Must know structure upfront
❌ **Performance**: JOINs slow at scale
❌ **Impedance mismatch**: SQL doesn't map cleanly to objects (ORM issues)

### When to Use SQL

**1. Complex Relationships**
- Many-to-many relationships
- Need JOINs across multiple tables
- Example: E-commerce (users, products, orders, order_items)

**2. ACID Compliance Required**
- Financial transactions (banking, payments)
- Inventory management
- Booking systems (prevent double-booking)

**3. Complex Queries**
- Aggregations: COUNT, SUM, AVG, GROUP BY
- Reporting and analytics
- Ad-hoc queries

**4. Data Integrity Critical**
- Foreign key constraints prevent orphaned data
- Check constraints validate data
- Example: User accounts, medical records

---

## NoSQL Databases in Detail

### Types of NoSQL

**1. Document Databases (MongoDB, CouchDB)**

Store JSON-like documents:
\`\`\`json
{
  "_id": "user123",
  "name": "Alice",
  "email": "alice@example.com",
  "address": {
    "street": "123 Main St",
    "city": "NYC"
  },
  "tags": ["premium", "verified"]
}
\`\`\`

**Use cases**: Content management, user profiles, catalogs

**2. Key-Value Stores (Redis, DynamoDB)**

Simple key-value pairs:
\`\`\`
Key: "session:abc123"
Value: {"user_id": 101, "expires": 1704067200}
\`\`\`

**Use cases**: Caching, session storage, real-time data

**3. Column-Family (Cassandra, HBase)**

Store data in columns, not rows:
\`\`\`
Row key: "user123"
Columns: {name: "Alice", email: "alice@example.com", ...}
\`\`\`

**Use cases**: Time-series data, IoT, analytics

**4. Graph Databases (Neo4j, Amazon Neptune)**

Store nodes and relationships:
\`\`\`
(Alice)-[:FRIENDS_WITH]->(Bob)
(Alice)-[:LIKES]->(Post123)
\`\`\`

**Use cases**: Social networks, recommendation engines, fraud detection

### BASE Properties

**Basically Available**: System always responds (even if stale)

**Soft state**: Data may change without input (due to eventual consistency)

**Eventually consistent**: All nodes converge to same state eventually

### Advantages

✅ **Schema flexibility**: Add fields without migrations
✅ **Horizontal scaling**: Designed to scale out (sharding built-in)
✅ **High performance**: Optimized for specific access patterns
✅ **High availability**: Multi-master, eventual consistency
✅ **Large scale**: Handle billions of records efficiently

### Disadvantages

❌ **No ACID transactions** (most NoSQL): Eventually consistent
❌ **No JOINs**: Must denormalize or do application-level joins
❌ **Limited query capability**: Can't do arbitrary queries
❌ **Data duplication**: Denormalization means redundancy
❌ **Less mature**: Fewer tools, less standardization

### When to Use NoSQL

**1. Massive Scale**
- Billions of records
- Horizontal scaling required
- Example: Social media (Facebook, Twitter)

**2. Flexible Schema**
- Data structure evolves frequently
- Different records have different fields
- Example: Content management, product catalogs with varied attributes

**3. Specific Access Patterns**
- Know queries upfront
- Optimize for specific queries
- Example: Real-time analytics, time-series data

**4. High Availability**
- Can't afford downtime
- Eventual consistency acceptable
- Example: Session storage, user profiles

---

## Real-World Examples

### Example 1: E-commerce System (Hybrid)

**SQL (PostgreSQL)**: Core transactional data
- Users table
- Orders table  
- Payments table
- Inventory table

**Why SQL**: ACID transactions critical (can't sell same item twice, can't lose payment)

**NoSQL (MongoDB)**: Product catalog
- Product details (varying attributes per category)
- User reviews
- Search index

**Why NoSQL**: Flexible schema (electronics have different attributes than clothing), read-heavy

**NoSQL (Redis)**: Caching and sessions
- Session storage
- Shopping cart (temporary data)
- Product cache

**Why NoSQL**: Fast access, temporary data, high read throughput

---

### Example 2: Social Media (Instagram)

**SQL (PostgreSQL)**: Core user data
- User accounts
- Relationships (followers/following)
- Authentication

**Why SQL**: Data integrity (user accounts must be accurate), complex queries (find mutual friends)

**NoSQL (Cassandra)**: Feed and posts
- User posts
- Timeline feeds
- Comments, likes

**Why NoSQL**: Massive scale (billions of posts), denormalized for performance, eventual consistency acceptable

**NoSQL (Redis)**: Real-time features
- Online status
- Notifications counter
- Rate limiting

**Why NoSQL**: Low latency (<1ms), ephemeral data

---

## SQL vs NoSQL Trade-offs

### Schema

**SQL**: 
- Fixed schema, must plan upfront
- Migrations complex at scale
- Example: Adding column requires ALTER TABLE (locks table)

**NoSQL**:
- Flexible schema, add fields anytime
- No migrations needed
- Example: New field in JSON document (instant)

### Scalability

**SQL**:
- Vertical scaling easy (more CPU/RAM)
- Horizontal scaling hard (sharding manual)
- Example: PostgreSQL write scaling requires complex sharding

**NoSQL**:
- Horizontal scaling built-in
- Add nodes, automatic rebalancing
- Example: Cassandra scales linearly (3 nodes → 30 nodes)

### Queries

**SQL**:
- Complex queries: JOINs, subqueries, aggregations
- Ad-hoc queries (explore data)
- Example: "Find top 10 customers who bought products in category X in last month"

**NoSQL**:
- Simple queries, specific access patterns
- Must know queries upfront
- Example: "Get all posts by user_id" (fast), but "Find all posts mentioning keyword X" (slow/impossible)

### Consistency

**SQL**:
- Strong consistency (ACID)
- Immediate consistency across all reads
- Example: Bank transfer (debit/credit atomic)

**NoSQL**:
- Eventual consistency (most NoSQL)
- Reads might return stale data briefly
- Example: Facebook like count (eventually accurate, brief staleness OK)

---

## Polyglot Persistence

Modern systems use **multiple databases** for different needs:

### Example: Netflix Architecture

**MySQL**: Billing, subscriptions (ACID required)

**Cassandra**: User viewing history, recommendations (scale)

**ElasticSearch**: Search functionality

**Redis**: Session cache

**Why**: Each database optimized for its use case. "Use the right tool for the job."

---

## Migration Between SQL and NoSQL

### Scenario: Growing Beyond SQL

**Problem**: PostgreSQL at capacity (sharding needed)

**Option 1**: Shard PostgreSQL
- Complex (manual sharding logic)
- Limited scalability
- Requires expertise

**Option 2**: Migrate to Cassandra
- Built-in sharding
- Better scalability
- Trade-off: Lose ACID transactions

**Decision**: Keep SQL for critical data (orders, payments), move read-heavy data (feeds, logs) to Cassandra

---

## Common Mistakes

### ❌ Mistake 1: NoSQL for Everything

**Problem**: Using MongoDB for financial transactions

**Why bad**: No ACID = risk of lost/double transactions

**Better**: Use SQL for transactions, NoSQL for other data

### ❌ Mistake 2: SQL Without Understanding Scale

**Problem**: Starting with SQL, hitting scaling limits at 10M users

**Why bad**: Sharding SQL is complex, expensive migration

**Better**: If expecting massive scale, start with NoSQL (or plan SQL sharding early)

### ❌ Mistake 3: Treating NoSQL Like SQL

**Problem**: Doing application-level JOINs in NoSQL

**Why bad**: Defeats purpose of NoSQL, slow performance

**Better**: Denormalize data for NoSQL access patterns

---

## Best Practices

### ✅ 1. Start with SQL Unless You Have Good Reason

SQL is simpler, well-understood, sufficient for most applications. Use NoSQL when you have specific needs (scale, flexibility, performance).

### ✅ 2. Use Polyglot Persistence

Don't choose "SQL vs NoSQL" - use both! SQL for transactional data, NoSQL for scale/flexibility.

### ✅ 3. Know Your Access Patterns

If using NoSQL, design schema based on queries (not normalized structure). Example: Denormalize user info into posts for fast timeline.

### ✅ 4. Test at Scale

SQL performs great at 1M records, but what about 1B? Load test before committing.

### ✅ 5. Consider Operations

SQL has mature tooling (backups, replication, monitoring). NoSQL may require more operational expertise.

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd recommend:

**SQL (PostgreSQL) for**:
- [Core transactional data: users, orders, payments]
- Reasoning: ACID transactions critical, data integrity required
- Trade-off: Harder to scale writes, but manageable with read replicas

**NoSQL (MongoDB) for**:
- [Product catalog, user-generated content]
- Reasoning: Flexible schema (products have varying attributes), read-heavy
- Trade-off: Eventual consistency, denormalization complexity

**NoSQL (Redis) for**:
- [Session storage, caching]
- Reasoning: Fast access (<1ms), temporary data
- Trade-off: In-memory (limited storage)

**Approach**: Polyglot persistence - use right database for each use case."

---

## Summary Table

| Aspect | SQL | NoSQL |
|--------|-----|-------|
| **Schema** | Fixed, predefined | Flexible, schema-less |
| **Scalability** | Vertical (hard to shard) | Horizontal (built-in sharding) |
| **Transactions** | ACID (strong consistency) | BASE (eventual consistency) |
| **Queries** | Complex (JOINs, aggregations) | Simple (key-based access) |
| **Data Model** | Normalized (reduce redundancy) | Denormalized (duplicate for performance) |
| **Use Cases** | Transactions, complex queries | Scale, flexibility, specific patterns |
| **Examples** | PostgreSQL, MySQL | MongoDB, Cassandra, Redis |
| **Best For** | Financial, inventory, booking | Social media, catalogs, IoT |

---

## Key Takeaways

✅ SQL: ACID transactions, complex queries, strong consistency
✅ NoSQL: Scale, flexibility, specific access patterns
✅ Use SQL for transactional, critical data (payments, inventory)
✅ Use NoSQL for massive scale, flexible schema (feeds, catalogs)
✅ Polyglot persistence: Use multiple databases for different needs
✅ SQL scales vertically, NoSQL scales horizontally
✅ NoSQL requires denormalization (application-level JOINs are anti-pattern)
✅ Start with SQL unless you have specific NoSQL requirements`,
};
