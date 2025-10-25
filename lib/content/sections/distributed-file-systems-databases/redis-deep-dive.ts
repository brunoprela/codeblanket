/**
 * Redis Deep Dive Section
 */

export const redisSection = {
  id: 'redis-deep-dive',
  title: 'Redis Deep Dive',
  content: `Redis (REmote DIctionary Server) is an in-memory data structure store used as a database, cache, message broker, and streaming engine, known for its blazing fast performance and versatile data structures.

## Overview

**Redis** = In-memory key-value data structure store

**Created**: 2009 by Salvatore Sanfilippo

**Key characteristics**:
- In-memory storage (optionally persistent)
- Sub-millisecond latency (<1ms typical)
- Rich data structures (not just strings!)
- Single-threaded (mostly)
- Atomic operations
- Pub/Sub messaging
- Lua scripting

**Used by**:
- Twitter (timeline caching)
- GitHub (job queues)
- Stack Overflow (caching)
- Snapchat (user data)

**Performance**:
- 100,000+ ops/sec per instance
- Sub-millisecond latency
- GBs to TBs in memory

---

## Data Structures

### 1. Strings

**Simple key-value**:

\`\`\`redis
SET user:123:name "Alice"
GET user:123:name  # Returns: "Alice"

# Atomic increment
INCR page:views  # Returns: 1
INCR page:views  # Returns: 2

# Set with expiration
SETEX session:abc123 3600 "user_data"  # Expires in 1 hour

# Set if not exists
SETNX lock:resource1 "locked"  # Returns 1 if successful, 0 if exists
\`\`\`

**Use cases**:
- Caching
- Session storage
- Counters
- Distributed locks

### 2. Hashes

**Field-value pairs** (like objects):

\`\`\`redis
HSET user:123 name "Alice" email "alice@example.com" age 30
HGET user:123 name  # Returns: "Alice"
HGETALL user:123    # Returns: {"name": "Alice", "email": "...", "age": "30"}

HINCRBY user:123 age 1  # Atomic increment field
HDEL user:123 email     # Delete field
\`\`\`

**Use cases**:
- User profiles
- Object caching
- Configuration storage

**Why use hash instead of string?**
- Update individual fields without fetching entire object
- Memory efficient for many small fields
- Atomic field operations

### 3. Lists

**Ordered collection** (doubly linked list):

\`\`\`redis
LPUSH queue:tasks "task1"  # Add to left (head)
RPUSH queue:tasks "task2"  # Add to right (tail)

LPOP queue:tasks  # Remove from left: "task1"
RPOP queue:tasks  # Remove from right: "task2"

# Blocking pop (wait until element available)
BLPOP queue:tasks 30  # Wait up to 30 seconds

# Range
LRANGE queue:tasks 0 9  # Get first 10 elements

# Trim (keep only range)
LTRIM queue:tasks 0 999  # Keep only first 1000 elements
\`\`\`

**Use cases**:
- Message queues
- Activity feeds
- Recent items
- Leaderboards (with ordered access)

### 4. Sets

**Unordered collection of unique strings**:

\`\`\`redis
SADD user:123:interests "reading" "travel" "photography"
SMEMBERS user:123:interests  # Returns: {"reading", "travel", "photography"}

SISMEMBER user:123:interests "reading"  # Returns: 1 (exists)

# Set operations
SINTER user:123:interests user:456:interests  # Intersection
SUNION user:123:interests user:456:interests  # Union
SDIFF user:123:interests user:456:interests   # Difference

SCARD user:123:interests  # Count: 3

SREM user:123:interests "travel"  # Remove element
\`\`\`

**Use cases**:
- Tags
- Unique visitors
- Social graphs (followers, following)
- Recommendation systems (common interests)

### 5. Sorted Sets (ZSets)

**Ordered by score**:

\`\`\`redis
# Add members with scores
ZADD leaderboard 1000 "player1" 1500 "player2" 800 "player3"

# Range by rank (0-based)
ZRANGE leaderboard 0 9  # Top 10 players (lowest to highest score)
ZREVRANGE leaderboard 0 9  # Top 10 (highest to lowest)

# Range by score
ZRANGEBYSCORE leaderboard 1000 2000  # Players with score 1000-2000

# Rank
ZRANK leaderboard "player1"      # Returns: 0 (lowest score)
ZREVRANK leaderboard "player2"   # Returns: 0 (highest score)

# Score
ZSCORE leaderboard "player1"  # Returns: "1000"

# Increment score
ZINCRBY leaderboard 100 "player1"  # player1 now has 1100

# Count
ZCARD leaderboard  # Returns: 3
\`\`\`

**Use cases**:
- Leaderboards
- Priority queues
- Time-series data (score = timestamp)
- Rate limiting (sliding window)

### 6. Bitmaps

**Bit-level operations on strings**:

\`\`\`redis
# Set bit
SETBIT user:123:login:2024-01-15 0 1  # User logged in

# Get bit
GETBIT user:123:login:2024-01-15 0  # Returns: 1

# Count bits set to 1
BITCOUNT user:123:login:2024-01-15  # Count login events

# Bitwise operations
BITOP AND result key1 key2  # AND operation
BITOP OR result key1 key2   # OR operation
\`\`\`

**Use cases**:
- Real-time analytics (daily active users)
- Feature flags
- Permissions
- Tracking events (very memory efficient!)

**Example**: Track daily active users
\`\`\`
Day 1: SETBIT dau:2024-01-15 123 1  (user 123 active)
Day 1: SETBIT dau:2024-01-15 456 1  (user 456 active)
Day 1: BITCOUNT dau:2024-01-15       (returns: 2 DAU)

Week total: BITOP OR dau:week dau:2024-01-15 dau:2024-01-16 ... dau:2024-01-21
            BITCOUNT dau:week  (returns: weekly active users)
\`\`\`

### 7. HyperLogLog

**Probabilistic cardinality estimation**:

\`\`\`redis
PFADD unique:visitors user1 user2 user3
PFCOUNT unique:visitors  # Returns: 3 (approximate)

# Merge multiple HyperLogLogs
PFMERGE result hll1 hll2 hll3
\`\`\`

**Use cases**:
- Unique visitors count
- Distinct elements (when exact count not needed)
- Memory efficient (12 KB for billions of elements!)

**Trade-off**: 0.81% standard error

### 8. Streams

**Append-only log** (like Kafka topics):

\`\`\`redis
# Add entry (auto-generated ID)
XADD events * user_id 123 action "login" timestamp 1705329600

# Read entries
XRANGE events - +  # Read all
XREAD STREAMS events 0  # Read from beginning

# Consumer groups
XGROUP CREATE events mygroup 0
XREADGROUP GROUP mygroup consumer1 STREAMS events >

# Acknowledge
XACK events mygroup <id>
\`\`\`

**Use cases**:
- Event sourcing
- Audit logs
- Message broker
- Time-series data

---

## Persistence

### 1. RDB (Redis Database)

**Point-in-time snapshots**:

\`\`\`redis
# Manual snapshot
SAVE   # Blocking
BGSAVE # Background (non-blocking)

# Automatic snapshots
save 900 1   # Save if at least 1 key changed in 900 sec
save 300 10  # Save if at least 10 keys changed in 300 sec
save 60 10000 # Save if at least 10000 keys changed in 60 sec
\`\`\`

**Benefits**:
- ✅ Compact single file
- ✅ Fast restarts
- ✅ Good for backups

**Drawbacks**:
- ❌ Can lose data between snapshots
- ❌ Fork can cause latency spike (large datasets)

### 2. AOF (Append Only File)

**Log every write operation**:

\`\`\`redis
# Enable AOF
appendonly yes

# Fsync policy
appendfsync always     # Fsync after every write (slowest, most durable)
appendfsync everysec   # Fsync every second (default)
appendfsync no         # Let OS decide (fastest, least durable)
\`\`\`

**Rewrite** (compaction):
\`\`\`redis
BGREWRITEAOF  # Compact AOF file
\`\`\`

**Benefits**:
- ✅ More durable (lose at most 1 second of data)
- ✅ Append-only (no corruption)
- ✅ Human-readable

**Drawbacks**:
- ❌ Larger files than RDB
- ❌ Slower restarts
- ❌ Can be slower than RDB

### 3. Hybrid (RDB + AOF)

**Best of both worlds**:
- RDB snapshot for fast restarts
- AOF for durability since last snapshot

**Recommended for production**

---

## Replication

### Master-Replica

\`\`\`
        Master (R/W)
         /      \\
        /        \\
  Replica 1    Replica 2
   (R only)    (R only)
\`\`\`

**Setup**:
\`\`\`redis
# On replica
REPLICAOF master-ip 6379

# On master
INFO replication  # See connected replicas
\`\`\`

**Replication**:
- Asynchronous (eventual consistency)
- Master sends commands to replicas
- Replicas replay commands

**Use cases**:
- Read scaling (read from replicas)
- High availability (promote replica on master failure)
- Backup (snapshot replica without impacting master)

---

## Sentinel (High Availability)

### Automatic Failover

\`\`\`
 Sentinel1   Sentinel2   Sentinel3
      \\        |        /
       \\       |       /
        \\      |      /
        Master ← Replica1 ← Replica2
\`\`\`

**Sentinel responsibilities**:
- Monitor master and replicas
- Detect master failure
- Automatic failover (promote replica)
- Notify clients of new master

**Quorum**: Number of Sentinels needed to agree on failure
- Minimum 3 Sentinels (avoid split-brain)
- Majority agreement to elect new master

---

## Redis Cluster

### Automatic Sharding

\`\`\`
   Client
      ↓
   Cluster
   /  |  \\
  /   |   \\
Node1 Node2 Node3
(Master) (Master) (Master)
  ↓     ↓     ↓
Replica Replica Replica
\`\`\`

**Sharding**:
- 16,384 hash slots
- Each key mapped to slot: \`CRC16(key) % 16384\`
- Each master owns subset of slots
- Example: Node1 (0-5460), Node2 (5461-10922), Node3 (10923-16383)

**Hash tags** (keep related keys on same node):
\`\`\`redis
SET user:{123}:profile "..."
SET user:{123}:settings "..."
# Both on same node (hash on "123")
\`\`\`

**Replication**:
- Each master has 1+ replicas
- Replica promoted if master fails

**No proxy needed**:
- Clients connect directly to nodes
- Node redirects if key on different node: \`MOVED\` or \`ASK\`

---

## Transactions

### MULTI/EXEC

\`\`\`redis
MULTI
SET account1:balance 900
SET account2:balance 1100
EXEC
# All or nothing (atomic)
\`\`\`

**Not true ACID**:
- No rollback on error
- Commands queued and executed atomically
- Isolation via single-threaded execution

### WATCH (Optimistic Locking)

\`\`\`redis
WATCH balance
val = GET balance
val = val - 100
MULTI
SET balance val
EXEC
# EXEC fails if balance was modified during WATCH
\`\`\`

---

## Lua Scripting

### Atomic Scripts

\`\`\`lua
-- Atomically get and increment
local value = redis.call('GET', KEYS[1])
redis.call('INCR', KEYS[1])
return value
\`\`\`

\`\`\`redis
EVAL "return redis.call('GET', KEYS[1])" 1 mykey
\`\`\`

**Benefits**:
- ✅ Atomic execution
- ✅ Reduce network round trips
- ✅ Complex logic server-side

---

## Pub/Sub

### Messaging

\`\`\`redis
# Subscribe
SUBSCRIBE news sports

# Publish
PUBLISH news "Breaking news!"
PUBLISH sports "Score update"

# Pattern subscribe
PSUBSCRIBE news:*
\`\`\`

**Fire-and-forget**:
- No persistence
- No guaranteed delivery
- Subscribers must be online

**Use cases**:
- Real-time notifications
- Chat applications
- Live updates

---

## Common Use Cases

### 1. Caching

\`\`\`python
def get_user (user_id):
    # Try cache
    cached = redis.get (f"user:{user_id}")
    if cached:
        return json.loads (cached)
    
    # Cache miss - fetch from DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache (1 hour TTL)
    redis.setex (f"user:{user_id}", 3600, json.dumps (user))
    return user
\`\`\`

### 2. Session Storage

\`\`\`python
# Store session
redis.setex (f"session:{session_id}", 1800, json.dumps (session_data))

# Get session
session = redis.get (f"session:{session_id}")

# Extend session
redis.expire (f"session:{session_id}", 1800)
\`\`\`

### 3. Rate Limiting

**Fixed window**:
\`\`\`python
key = f"rate_limit:{user_id}:{current_minute}"
count = redis.incr (key)
redis.expire (key, 60)

if count > 100:
    raise RateLimitExceeded()
\`\`\`

**Sliding window** (with sorted set):
\`\`\`python
now = time.time()
window = 60  # 60 seconds
key = f"rate_limit:{user_id}"

# Remove old entries
redis.zremrangebyscore (key, 0, now - window)

# Count requests in window
count = redis.zcard (key)

if count >= 100:
    raise RateLimitExceeded()

# Add current request
redis.zadd (key, {str (uuid.uuid4()): now})
redis.expire (key, window)
\`\`\`

### 4. Leaderboard

\`\`\`python
# Update score
redis.zincrby("leaderboard", 100, f"player:{player_id}")

# Get top 10
top_10 = redis.zrevrange("leaderboard", 0, 9, withscores=True)

# Get player rank
rank = redis.zrevrank("leaderboard", f"player:{player_id}")

# Get player score
score = redis.zscore("leaderboard", f"player:{player_id}")
\`\`\`

### 5. Distributed Lock

\`\`\`python
# Acquire lock
acquired = redis.set(
    "lock:resource",
    "unique_id",
    nx=True,  # Set if not exists
    ex=30     # Expire in 30 seconds
)

if acquired:
    try:
        # Critical section
        process()
    finally:
        # Release lock (with Lua for atomicity)
        redis.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end",
            1, "lock:resource", "unique_id"
        )
\`\`\`

---

## Performance Best Practices

### 1. Use Pipelining

**Bad** (N round trips):
\`\`\`python
for key in keys:
    redis.get (key)
\`\`\`

**Good** (1 round trip):
\`\`\`python
pipe = redis.pipeline()
for key in keys:
    pipe.get (key)
results = pipe.execute()
\`\`\`

### 2. Use Connection Pooling

\`\`\`python
pool = redis.ConnectionPool (host='localhost', port=6379, max_connections=50)
redis_client = redis.Redis (connection_pool=pool)
\`\`\`

### 3. Avoid KEYS Command

**Bad** (blocks Redis):
\`\`\`redis
KEYS user:*  # Scans all keys (O(N))
\`\`\`

**Good** (non-blocking):
\`\`\`redis
SCAN 0 MATCH user:* COUNT 100  # Iterates incrementally
\`\`\`

### 4. Set Expiration

**Always set TTL** to avoid memory bloat:
\`\`\`redis
SETEX cache:key 3600 "value"
\`\`\`

### 5. Use Appropriate Data Structure

- Use hash for objects (not multiple strings)
- Use sorted set for rankings (not lists)
- Use HyperLogLog for large cardinality

---

## Monitoring

**Key metrics**:
\`\`\`redis
INFO stats
INFO memory
INFO replication

# Live monitoring
MONITOR  # See all commands in real-time (expensive!)

# Slow log
SLOWLOG GET 10  # Get 10 slowest commands
\`\`\`

**Critical metrics**:
- Memory usage
- Hit rate (keyspace hits vs misses)
- Connected clients
- Commands per second
- Evicted keys (memory pressure)

---

## Interview Tips

**Explain Redis in 2 minutes**:
"Redis is an in-memory data structure store providing sub-millisecond latency. It\'s not just key-value - supports strings, hashes, lists, sets, sorted sets, bitmaps, HyperLogLog, and streams. Single-threaded for simplicity and atomicity. Persistence via RDB snapshots or AOF log. Replication for read scaling and HA. Sentinel for automatic failover. Redis Cluster for horizontal scaling with automatic sharding. Common use cases: caching, session storage, rate limiting, leaderboards, pub/sub, distributed locks. All operations are atomic. Lua scripting for complex atomic operations."

**Key trade-offs**:
- In-memory vs persistence: Speed vs durability
- RDB vs AOF: Compact/fast vs durable
- Replication: Eventual consistency
- Cluster: Complexity vs scalability

**Common mistakes**:
- ❌ Not setting TTL (memory bloat)
- ❌ Using KEYS instead of SCAN
- ❌ Not using pipelining for batch operations
- ❌ Treating as durable database (it's a cache!)

---

## Key Takeaways

🔑 In-memory data structure store (sub-ms latency)
🔑 Rich data structures: strings, hashes, lists, sets, sorted sets, bitmaps, HyperLogLog, streams
🔑 Persistence: RDB (snapshots) and AOF (append-only log)
🔑 Replication: async master-replica
🔑 Sentinel: automatic failover
🔑 Cluster: automatic sharding across nodes
🔑 All operations atomic (single-threaded)
🔑 Common uses: caching, sessions, rate limiting, leaderboards, pub/sub, distributed locks
🔑 Pipeline for batch operations
🔑 Best for: Speed > durability, in-memory data
`,
};
