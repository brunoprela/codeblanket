/**
 * Data Partitioning & Sharding Section
 */

export const datapartitioningshardingSection = {
  id: 'data-partitioning-sharding',
  title: 'Data Partitioning & Sharding',
  content: `Data partitioning (sharding) splits large databases into smaller, faster, more manageable pieces called shards to achieve horizontal scalability.

## What is Data Partitioning?

**Definition**: Splitting a large dataset across multiple databases or servers so that each machine holds only a portion of the data.

### **Why Partition Data?**

**Without Partitioning:**
- Single database has limits (storage, CPU, memory, connections)
- Query performance degrades as data grows
- Single point of failure
- Can't scale beyond one machine's capacity

**Example**: Twitter has billions of tweets. Can't fit all tweets in a single PostgreSQL database.

**With Partitioning:**
- Horizontal scalability (add more machines)
- Better query performance (each shard smaller, queries faster)
- Higher throughput (queries distributed across shards)
- Fault isolation (one shard fails, others still work)

**Real-world scale**: Instagram shards photos across thousands of database servers.

---

## Horizontal vs Vertical Partitioning

### **Vertical Partitioning**

**Definition**: Split table by columns.

**Example**: User table with 50 columns
- Shard 1: \`id, username, email\` (frequently accessed)
- Shard 2: \`profile_description, hobbies, bio\` (rarely accessed)

**Use case**: Separate hot columns (accessed often) from cold columns (accessed rarely).

**Pros:**
- Reduces I/O for queries accessing only hot columns
- Simpler queries (fewer joins)

**Cons:**
- Still limited by single machine (not true horizontal scaling)
- Joins across shards more complex

**This is less common than horizontal partitioning.**

---

### **Horizontal Partitioning (Sharding)**

**Definition**: Split table by rows.

**Example**: User table with 100M users
- Shard 1: Users 1-25M
- Shard 2: Users 25M-50M
- Shard 3: Users 50M-75M
- Shard 4: Users 75M-100M

**Use case**: Scale beyond single machine capacity (most common).

**Pros:**
- True horizontal scalability (add more shards)
- Each shard smaller and faster
- Parallelism (multiple queries across shards simultaneously)

**Cons:**
- Complex queries (cross-shard joins, aggregations)
- Rebalancing when adding shards
- Application logic more complex

**This is what most people mean by "sharding."**

---

## Sharding Strategies

### **1. Range-Based Sharding**

**How it works**: Partition data based on ranges of a key (e.g., user ID, date).

**Example: Shard by User ID**
- Shard 1: User IDs 1-1,000,000
- Shard 2: User IDs 1,000,001-2,000,000
- Shard 3: User IDs 2,000,001-3,000,000

**Example: Shard by Date**
- Shard 1: Orders from 2020
- Shard 2: Orders from 2021
- Shard 3: Orders from 2022
- Shard 4: Orders from 2023

**Pros:**
- Simple to implement
- Range queries efficient (all data in one or few shards)
- Easy to understand

**Cons:**
- **Hotspots**: Uneven distribution if data not uniform
  - Example: New users always go to latest shard (Shard 3 overloaded, Shard 1 idle)
- **Sequential IDs**: If user IDs sequential, recent users on one shard (hotspot)

**When to use**: When data has natural ranges and distribution is even.

---

### **2. Hash-Based Sharding**

**How it works**: Apply hash function to partition key, use result to determine shard.

**Formula**: \`shard_id = hash (key) % num_shards\`

**Example: Shard by User ID**
- User ID 123 → \`hash(123) % 4\` = 3 → Shard 3
- User ID 456 → \`hash(456) % 4\` = 0 → Shard 0
- User ID 789 → \`hash(789) % 4\` = 1 → Shard 1

**Pros:**
- **Uniform distribution**: Hash function distributes data evenly
- **No hotspots**: Random distribution prevents any shard from being overloaded
- Simple to implement

**Cons:**
- **Range queries hard**: Finding users 1-1000 requires querying ALL shards
- **Rebalancing**: Adding/removing shards requires rehashing all data
  - Example: 4 shards → 5 shards changes all \`hash (key) % 4\` to \`hash (key) % 5\`
  - Requires massive data migration

**When to use**: When uniform distribution more important than range queries.

---

### **3. Consistent Hashing**

**How it works**: Hash both data and shards onto a ring (0 to 2^32-1). Data assigned to next shard clockwise on ring.

**Visual**:

**Hash Ring Visualization:**
- Ring ranges from 0 to 2^32-1
- Shards: S1 at 25%, S2 at 50%, S3 at 75%
- User 123: hash(123) = 30% → Next shard clockwise = S2
- User 456: hash(456) = 60% → Next shard clockwise = S3

**Adding a shard**:
- New Shard S4 at 40%
- Only data between 25% and 40% moves (from S2 to S4)
- Other shards unaffected

**Virtual Nodes**: Place each physical shard at multiple positions on ring for better distribution.

**Pros:**
- **Minimal rebalancing**: Adding/removing shards only affects neighboring data
- **Scalable**: Easy to add shards
- **Uniform distribution** with virtual nodes

**Cons:**
- More complex to implement
- Still no efficient range queries

**Use case**: Distributed caches (Redis Cluster), distributed databases (Cassandra, DynamoDB).

**This is the standard for modern distributed systems!**

---

### **4. Directory-Based Sharding**

**How it works**: Maintain a lookup table (directory) mapping keys to shards.

**Example**:
\`\`\`
Directory Table:
User ID Range → Shard
    1 - 1M       → Shard 1
    1M - 2M      → Shard 2
    2M - 3M      → Shard 3
        \`\`\`

**Lookup**:
- User ID 500,000 → Check directory → Shard 1
- User ID 1,500,000 → Check directory → Shard 2

**Pros:**
- **Flexible**: Can rebalance by updating directory (no data migration immediately)
- **Custom logic**: Can shard based on complex rules
- **Dynamic**: Easy to add shards, update mappings

**Cons:**
- **Directory is bottleneck**: Every query must lookup directory first
- **Single point of failure**: If directory down, entire system down
- **Latency**: Extra hop for every query

**Mitigation**: Cache directory, replicate directory.

**When to use**: When need flexibility and custom sharding logic.

---

### **5. Geo-Based Sharding**

**How it works**: Partition data by geographic region.

**Example**:
- Shard 1 (US-East): Users in US-East coast
- Shard 2 (US-West): Users in US-West coast
- Shard 3 (EU): Users in Europe
- Shard 4 (Asia): Users in Asia

**Pros:**
- **Reduced latency**: Data stored close to users
- **Compliance**: Data sovereignty laws (EU data must stay in EU)
- **Natural partitioning**: Users mostly access local data

**Cons:**
- **Uneven distribution**: Some regions have more users
- **Cross-region queries**: Complex and slow

**When to use**: Global applications with geographic user distribution.

---

## Choosing a Partition Key

**Partition key** determines how data is distributed across shards.

### **Good Partition Key Characteristics:**1. **High cardinality**: Many unique values (good: user ID, order ID; bad: country, gender)
2. **Uniform distribution**: Values evenly spread (good: hash of email; bad: sequential IDs)
3. **Predictable access patterns**: Key used in most queries (good: user_id for user queries)

### **Examples:**

**Good**: \`user_id\` for user data
- High cardinality (millions of users)
- Uniform distribution (with hash)
- Queries typically filter by user_id

**Bad**: \`country\` for user data
- Low cardinality (~200 countries)
- Uneven distribution (US has more users than Liechtenstein)
- Creates hotspots

**Bad**: \`created_at\` (timestamp) for logs
- Recent data all goes to one shard (hotspot)
- Old shards idle

**Better for logs**: \`hash (log_id)\` or combine \`created_at\` + \`hash (source_id)\`

---

## Cross-Shard Operations

### **Problem: Joins Across Shards**

**Example**: Users sharded by \`user_id\`, Posts sharded by \`post_id\`.

Query: "Get all posts by user 123 with >100 likes"

**Challenge**: 
- User 123 on Shard 2
- Posts scattered across all shards

**Solution approaches:**

**1. Denormalization**: Store user info with each post (duplicate data, avoid join)

**2. Application-level joins**: Query all shards, merge results in application

**3. Shard by same key**: Shard users AND posts by \`user_id\` (colocate related data)

**Recommendation**: Design schema to avoid cross-shard joins. Denormalize or shard by dominant access pattern.

---

### **Problem: Aggregations Across Shards**

**Example**: "Count total users"

**Challenge**: Users distributed across 10 shards.

**Solution**: Map-Reduce pattern
- **Map**: Each shard counts its users
- **Reduce**: Sum counts from all shards

**Implementation**:

**Map-Reduce Example:**
- Shard 1: 1M users
- Shard 2: 1.2M users
- Shard 3: 900K users
- ...
- Total: 1M + 1.2M + 900K + ... = 15M users

**Cost**: O(num_shards) queries.

---

### **Problem: Distributed Transactions**

**Example**: Transfer money from user A (Shard 1) to user B (Shard 2).

**Challenge**: Need atomicity across shards (both succeed or both fail).

**Solution approaches:**

**1. Avoid distributed transactions**: Design to avoid (e.g., use event sourcing)

**2. Two-Phase Commit (2PC)**:
- Phase 1: Ask all shards "Can you commit?"
- Phase 2: If all yes, commit; if any no, rollback

**Cons**: Slow, complex, blocks on failure

**3. Saga pattern**: Sequence of local transactions with compensating actions

**Recommendation**: Design to avoid distributed transactions. They're slow and complex.

---

## Rebalancing Shards

**Scenario**: Add a new shard (4 shards → 5 shards).

### **Problem with Simple Hash**

Old: \`shard_id = hash (key) % 4\`  
New: \`shard_id = hash (key) % 5\`

**Result**: Almost ALL data needs to move (hash values change).

**Example**:
- User 123: \`hash(123) % 4\` = 3 → Shard 3
- After adding Shard 5: \`hash(123) % 5\` = 2 → Shard 2 (moves!)

**This is why consistent hashing is better!**

---

### **Rebalancing Strategies**

**1. Stop-the-world**: Stop application, migrate data, restart
- **Pro**: Simple
- **Con**: Downtime

**2. Dual-write**: Write to both old and new shards during migration
- **Pro**: No downtime
- **Con**: Complex

**3. Consistent hashing**: Only move data from neighboring shards
- **Pro**: Minimal data movement
- **Con**: Requires consistent hashing implementation

---

## Handling Hotspots

**Problem**: One shard receives disproportionate traffic.

**Example**: Celebrity user with millions of followers on Shard 3. All requests for that user hit Shard 3.

### **Solutions:**

**1. Further partition hot shard**: Split Shard 3 into Shard 3a and 3b

**2. Replicate hot data**: Cache celebrity user's data in Redis (read-heavy workload)

**3. Composite partition key**: Instead of just \`user_id\`, use \`hash (user_id + timestamp)\` to distribute

**4. Dedicated shard for hot entities**: Move celebrity users to separate, more powerful shard

---

## Real-World Sharding Examples

### **Instagram**

**Challenge**: Billions of photos, can't fit in single database.

**Solution**: Shard by \`photo_id\`
- Generate unique photo ID with embedded shard ID
- Each photo stored on specific shard based on ID
- Consistent hashing for even distribution

**ID format**: \`[shard_id][timestamp][sequence]\`

---

### **Twitter**

**Challenge**: Billions of tweets, high write throughput.

**Solution**: Shard by \`user_id\`
- User\'s tweets stored together on same shard (good for "get user timeline" query)
- Snowflake ID generation (distributed, k-sorted IDs)

**Trade-off**: Cross-user queries (follower timelines) require fan-out to multiple shards.

---

### **Uber**

**Challenge**: Millions of trips, geographic distribution.

**Solution**: Geo-based sharding
- Shard by city/region
- Trips in San Francisco → Shard SF
- Trips in New York → Shard NY

**Benefit**: Reduced latency (data close to users), compliance with local laws.

---

## Interview Tips

### **Common Questions:**

**Q: "Your database has 100M users and can't handle the load. How would you scale it?"**

✅ Good answer:
"I'd implement sharding:
1. **Partition key**: \`user_id\` (high cardinality, queries filter by user_id)
2. **Strategy**: Consistent hashing (minimal rebalancing when adding shards)
3. **Initial shards**: 10 shards (10M users each)
4. **Rebalancing**: As data grows, add shards incrementally
5. **Trade-off**: Cross-user queries (e.g., 'find all users in city X') become harder, would need to query all shards or maintain separate index"

**Q: "What\'s the difference between sharding and replication?"**

✅ Good answer:
"**Sharding (horizontal partitioning)**: Splits data across machines. Each machine holds DIFFERENT data. Goal: Scale capacity and throughput.

**Replication**: Copies SAME data to multiple machines. Each machine holds same data. Goal: Availability and read scalability.

Often used together: Shard for writes, replicate each shard for reads."

**Q: "How do you handle a celebrity user with millions of followers (hotspot)?"**

✅ Good answer:
"Several approaches:
1. **Cache**: Cache celebrity's data in Redis (read-heavy workload)
2. **Read replicas**: Add more read replicas for shard containing celebrity
3. **Separate shard**: Move hot entities to dedicated, more powerful shard
4. **CDN**: Cache celebrity's public data at edge
5. **Rate limiting**: Limit requests per user to prevent abuse"

---

## Key Takeaways

1. **Sharding = horizontal partitioning**: Splits data across machines by rows
2. **Partition key critical**: Choose high-cardinality, uniformly distributed key
3. **Hash-based sharding**: Uniform distribution, but rebalancing hard
4. **Consistent hashing**: Industry standard, minimal rebalancing when adding shards
5. **Avoid cross-shard joins**: Denormalize or colocate related data
6. **Hotspots**: Monitor and handle (caching, replication, repartitioning)
7. **Rebalancing**: Plan ahead, use consistent hashing to minimize data movement`,
};
