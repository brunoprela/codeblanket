/**
 * In-Memory vs Persistent Storage Section
 */

export const inmemoryvspersistentstorageSection = {
  id: 'in-memory-vs-persistent-storage',
  title: 'In-Memory vs Persistent Storage',
  content: `Choosing between in-memory and persistent storage is a fundamental trade-off between speed and durability. Understanding when to use each is critical for system design.

## Definitions

**In-Memory Storage**:
- **Data stored in RAM** (Random Access Memory)
- Extremely fast (microseconds)
- **Volatile**: Data lost on restart/crash
- Examples: Redis, Memcached, application caches

**Persistent Storage**:
- **Data stored on disk** (HDD/SSD)
- Slower (milliseconds)
- **Durable**: Data survives restarts/crashes
- Examples: PostgreSQL, MySQL, MongoDB, S3

---

## In-Memory Storage in Detail

### How It Works

Data stored directly in server RAM, no disk I/O.

\`\`\`
Application → RAM (In-Memory) → Response (microseconds)
\`\`\`

### Use Cases

**1. Caching**

**Example**: Cache product details in Redis
- Database query: 50ms
- Redis lookup: 1ms
- 50x faster!

**2. Session Storage**

**Example**: Store user session tokens
- Fast authentication (< 1ms)
- Expire sessions automatically (TTL)

**3. Rate Limiting**

**Example**: Track API calls per user
- Increment counter in Redis
- Check limit before processing request

**4. Real-Time Analytics**

**Example**: Live leaderboard
- Update scores in Redis sorted set
- Query top 100 in < 1ms

### Advantages

✅ **Extremely fast**: 1000x faster than disk
✅ **Low latency**: Sub-millisecond response
✅ **High throughput**: Millions of operations/second
✅ **Simple data structures**: Key-value, lists, sets

### Disadvantages

❌ **Volatile**: Data lost on restart (unless persistence enabled)
❌ **Limited capacity**: RAM expensive (256GB typical, vs 10TB disk)
❌ **Higher cost**: $10/GB (RAM) vs $0.10/GB (disk)
❌ **No complex queries**: Limited to simple lookups

---

## Persistent Storage in Detail

### How It Works

Data written to disk, survives restarts.

\`\`\`
Application → Disk (Database) → Response (milliseconds)
\`\`\`

### Use Cases

**1. Primary Data Store**

**Example**: User accounts, orders, transactions
- Must not lose data
- ACID guarantees (PostgreSQL)

**2. Long-Term Storage**

**Example**: Historical logs, archives
- Retain for years
- Cheap storage (S3)

**3. Complex Queries**

**Example**: Analytics queries
- JOINs, aggregations, filtering
- SQL databases excel here

### Advantages

✅ **Durable**: Data survives crashes
✅ **Large capacity**: Terabytes to petabytes
✅ **Lower cost**: 100x cheaper per GB than RAM
✅ **Complex queries**: SQL, indexing, transactions

### Disadvantages

❌ **Slower**: 100-1000x slower than RAM
❌ **I/O bottleneck**: Disk seeks, write amplification
❌ **Higher latency**: Milliseconds vs microseconds

---

## Real-World Examples

### Example 1: E-Commerce Product Page

**Hybrid approach**:

**Persistent storage (PostgreSQL)**:
- Product details, inventory, prices
- Source of truth
- Durable, must not lose data

**In-memory cache (Redis)**:
- Cache top 1000 products
- 99% of traffic hits cache
- Reduces database load by 100x

**Flow**:
1. User requests product
2. Check Redis cache
3. If hit: Return from cache (1ms)
4. If miss: Query database (50ms), cache result
5. Next request: Cache hit (1ms)

**Result**: Fast user experience + data durability

---

### Example 2: Banking Application

**Persistent storage ONLY**:

**Why no cache**:
- Account balances: Must be 100% accurate
- Transactions: Must be durable (ACID)
- Cannot risk stale data (cache inconsistency)

**Implementation**:
- PostgreSQL with replication
- Write ahead log (WAL) for durability
- Read replicas for scaling reads

**Trade-off**: Slower (10-50ms) but guaranteed correctness

---

### Example 3: Social Media Feed

**Hybrid approach**:

**Persistent storage (Cassandra)**:
- All posts, comments, likes
- Distributed, durable

**In-memory cache (Redis)**:
- User\'s home feed (last 100 posts)
- Recently viewed posts
- Reduces latency from 50ms to 5ms

**Eventual consistency acceptable**:
- Seeing post 5 seconds late is fine
- Speed > perfect consistency

---

## Caching Strategies

### Cache-Aside (Lazy Loading)

Application manages cache explicitly.

\`\`\`python
def get_user (user_id):
    # Check cache
    user = redis.get (f"user:{user_id}")
    if user:
        return user  # Cache hit
    
    # Cache miss: Query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Cache for future
    redis.set (f"user:{user_id}", user, ttl=3600)  # 1 hour
    return user
\`\`\`

**Advantages**: Simple, cache only what's requested

**Disadvantages**: Cache miss penalty (database query)

---

### Write-Through

Write to cache and database simultaneously.

\`\`\`python
def update_user (user_id, data):
    # Update database
    db.update("UPDATE users SET ... WHERE id = ?", user_id, data)
    
    # Update cache
    redis.set (f"user:{user_id}", data, ttl=3600)
\`\`\`

**Advantages**: Cache always up-to-date

**Disadvantages**: Slower writes (two operations)

---

### Write-Behind (Write-Back)

Write to cache immediately, asynchronously write to database.

\`\`\`python
def update_user (user_id, data):
    # Write to cache (fast)
    redis.set (f"user:{user_id}", data, ttl=3600)
    
    # Queue database write (async)
    queue.publish("db_writes", {"user_id": user_id, "data": data})
    
    # Worker processes queue and writes to database
\`\`\`

**Advantages**: Fast writes, deferred database load

**Disadvantages**: Risk of data loss (if cache fails before database write)

---

## Cache Eviction Policies

When cache is full, which data to evict?

**LRU (Least Recently Used)**:
- Evict data not accessed recently
- Most common (Redis default)
- Good for workloads with temporal locality

**LFU (Least Frequently Used)**:
- Evict data accessed least often
- Good for workloads with popular items

**TTL (Time To Live)**:
- Evict data after expiration time
- Good for time-sensitive data (sessions, rate limits)

---

## Redis Persistence Options

Redis is in-memory but can persist to disk.

### RDB (Snapshotting)

Periodic snapshots of data to disk.

\`\`\`
save 900 1       # Save if 1 key changed in 900 seconds
save 300 10      # Save if 10 keys changed in 300 seconds
save 60 10000    # Save if 10000 keys changed in 60 seconds
\`\`\`

**Advantages**: Fast, compact file

**Disadvantages**: Data loss between snapshots (up to 15 minutes)

---

### AOF (Append Only File)

Log every write operation to file.

\`\`\`
appendonly yes
appendfsync everysec  # Sync to disk every second
\`\`\`

**Advantages**: Minimal data loss (1 second max)

**Disadvantages**: Larger files, slower restarts

---

### Hybrid (RDB + AOF)

Best of both: Fast snapshots + minimal data loss.

---

## Trade-off Analysis

### Speed

**In-memory**: 0.1-1ms
**Persistent**: 10-100ms

**Example**: 
- Redis GET: 0.5ms
- PostgreSQL SELECT: 50ms
- **100x difference!**

---

### Durability

**In-memory**: Data lost on restart
**Persistent**: Data survives crashes

**Critical for**:
- Financial transactions → Persistent
- User sessions → In-memory acceptable

---

### Capacity

**In-memory**: 256GB typical
**Persistent**: 10TB+ typical

**Cost**:
- 256GB RAM: $2,000/month (AWS)
- 10TB SSD: $1,000/month (AWS)

---

### Query Complexity

**In-memory**: Simple lookups (key-value)
**Persistent**: Complex queries (SQL JOINs, aggregations)

**Example**:
- "Get user by ID" → In-memory perfect
- "Find all users who purchased X in last 30 days" → Persistent (SQL)

---

## Best Practices

### ✅ 1. Use Hybrid Approach

Persistent for source of truth, in-memory for performance.

### ✅ 2. Cache Hot Data

Identify 20% of data that's accessed 80% of time (Pareto principle).

### ✅ 3. Set Appropriate TTLs

- User sessions: 1 hour
- Product catalog: 5 minutes
- User profile: 1 day

### ✅ 4. Monitor Cache Hit Rate

Target: >90% hit rate

**If < 90%**: Cache too small or TTL too short

### ✅ 5. Handle Cache Failures Gracefully

Cache down? Fall back to database (degraded but functional).

---

## Anti-Patterns

### ❌ 1. Caching Everything

**Problem**: Cache pollution, low hit rate

**Better**: Cache only hot data

---

### ❌ 2. No Cache Invalidation

**Problem**: Stale data

**Better**: Set TTLs, explicit invalidation on updates

---

### ❌ 3. Using Cache as Primary Store

**Problem**: Data loss on cache restart

**Better**: Persistent storage as source of truth

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd use:

**Persistent storage (PostgreSQL) for**:
- [User accounts, orders, transactions]
- Reasoning: Requires durability, ACID guarantees
- Trade-off: Slower (50ms), but data safety critical

**In-memory cache (Redis) for**:
- [Product catalog, user sessions, rate limiting]
- Reasoning: Extremely fast (<1ms), reduces database load
- Trade-off: Volatile, but acceptable for this data

**Caching strategy**:
- Cache-aside for reads
- Write-through for critical data
- Monitor cache hit rate (target >90%)

**Overall**: Hybrid approach balances speed and durability."

---

## Summary Table

| Aspect | In-Memory | Persistent |
|--------|-----------|------------|
| **Speed** | 0.1-1ms | 10-100ms |
| **Durability** | Volatile | Durable |
| **Capacity** | 256GB | 10TB+ |
| **Cost/GB** | $10 | $0.10 |
| **Queries** | Simple (key-value) | Complex (SQL) |
| **Use Cases** | Cache, sessions, rate limiting | Primary data, archives, analytics |
| **Examples** | Redis, Memcached | PostgreSQL, MySQL, S3 |

---

## Key Takeaways

✅ In-memory: 100x faster, volatile, limited capacity, high cost/GB
✅ Persistent: Durable, large capacity, low cost/GB, slower
✅ Use hybrid: Persistent for source of truth, in-memory for performance
✅ Cache hot data (20% of data = 80% of traffic)
✅ Redis persistence: RDB (snapshots) or AOF (append log)
✅ Monitor cache hit rate (target >90%)
✅ Never use cache as primary store (data loss risk)`,
};
