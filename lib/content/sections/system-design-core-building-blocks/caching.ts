/**
 * Caching Section
 */

export const cachingSection = {
  id: 'caching',
  title: 'Caching',
  content: `Caching stores frequently accessed data in a fast storage layer to reduce latency, database load, and improve system performance.

## What is Caching ?

** Definition **: A cache is a high - speed data storage layer that stores a subset of data so future requests for that data are served faster.

### ** Why Caching Matters **

** Without Cache:**
    - Every request hits the database(slow)
        - Database becomes bottleneck at scale
            - High latency for users
                - Expensive(database costs scale with requests)

** With Cache:**
    - Most requests served from memory(fast)
        - Database load reduced by 80 - 95 %
            - Lower latency(milliseconds vs hundreds of milliseconds)
                - Cost savings(cache cheaper than database scaling)

                    ** Real - world impact:**
- ** Reddit **: 99 % cache hit rate on homepage
    - ** Twitter **: Caches timelines to serve billions of requests
        - ** Netflix **: Caches video metadata and recommendations

---

## Cache Hit vs Cache Miss

### ** Cache Hit **

    Request data exists in cache → return immediately from cache.

** Example:**
    1. User requests profile for user ID 123
2. Check cache: Found! → Return from cache(2ms)
3. No database query needed

---

### ** Cache Miss **

    Request data NOT in cache → fetch from database, store in cache, return to user.

** Example:**
    1. User requests profile for user ID 456
2. Check cache: Not found(miss)
3. Query database(50ms)
4. Store result in cache
5. Return to user

    ** Subsequent requests for user ID 456 will be cache hits! **

        ---

### ** Cache Hit Rate **

** Formula:** \`Cache Hit Rate = Hits / (Hits + Misses)\`

    ** Example:**
        - 90 requests served from cache(hits)
            - 10 requests went to database(misses)
                - Hit rate: 90 / (90 + 10) = 90 %

** Target hit rates:**
- ** Good **: 80 - 90 %
- ** Excellent **: 95 %+
- ** Needs tuning **: <80%

** Higher hit rate = better performance and lower cost **

    ---

## Where to Place Cache

### ** 1. Client - Side Cache(Browser) **

** Location **: User's browser

    ** What's cached**: HTML pages, CSS, JavaScript, images

        ** Example **: HTTP cache headers
            - \`Cache - Control: max - age=3600\`(cache for 1 hour)
    - \`ETag\` for validation

        ** Pros:**
            - Fastest(no network request)
            - Reduces server load
                - Works offline

                    ** Cons:**
                        - No control after deployment(can't invalidate)
                            - Limited storage
                        - User can clear cache

---

### ** 2. CDN Cache(Edge) **

** Location **: Geographically distributed edge servers(CloudFront, Akamai)

                        ** What's cached**: Static assets (images, videos, CSS, JS)

                        ** Example **: User in Japan requests image
                        - First request: Fetches from US origin server(200ms)
                        - CDN caches at Tokyo edge location
                        - Subsequent requests: Served from Tokyo(20ms)

                        ** Pros:**
                        - Reduced latency(geographically close to users)
                        - Offloads origin server
                        - Scales globally

                        ** Cons:**
                        - Only for static / semi - static content
                            - CDN costs
    - Invalidation complexity

---

### ** 3. Application Cache(In - Memory) **

** Location **: Application server's memory

    ** What's cached**: Application-level data (local to each server)

        ** Example **: Each API server caches configuration in memory

            ** Pros:**
                - Extremely fast(no network call)
                    - Simple to implement

                        ** Cons:**
                            - Cache inconsistency across servers
                                - Lost on server restart
                                    - Limited by server memory

                                        ** Use case:** Small, read - only data that rarely changes(config, feature flags)

---

### ** 4. Distributed Cache(Redis, Memcached) **

** Location **: Dedicated cache cluster, separate from app servers

    ** What's cached**: Frequently accessed data (user sessions, API responses, database queries)

        ** Example:** Redis cluster shared by all API servers

            ** Pros:**
                - Shared across all app servers(consistent)
                    - Survives app server restarts
                        - Scales independently
                            - Rich features(TTL, data structures)

                                ** Cons:**
                                    - Network latency(1 - 2ms)
                                        - Additional infrastructure to manage
                                            - Cache becomes dependency

                                                ** This is the most common caching strategy for distributed systems! **

                                                    ---

### ** 5. Database Cache(Query Result Cache) **

** Location **: Inside database(MySQL query cache, PostgreSQL shared buffers)

    ** What's cached**: Query results

        ** Example **: MySQL caches\`SELECT * FROM users WHERE id = 123\`

            ** Pros:**
                - Transparent(application doesn't need to manage)
                    - Works automatically

                ** Cons:**
                - Limited control
                - Invalidation can be tricky
                - Often disabled in modern databases(MySQL 8.0 removed query cache)

                ** Recommendation **: Use application - level cache(Redis) instead for better control.

---

## Cache Reading Patterns

### ** 1. Cache - Aside(Lazy Loading) **

** Most common pattern.**

** Flow:**
    1. Application checks cache for data
2. If ** cache hit **: Return data from cache
3. If ** cache miss **:
- Query database
    - Store result in cache
        - Return data

            ** Pseudocode:**
\`\`\`python
def get_user(user_id):
    # Check cache first
    user = cache.get(f"user:{user_id}")
    if user:
        return user  # Cache hit
    
    # Cache miss: query database
    user = database.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache for next time
    cache.set(f"user:{user_id}", user, ttl=3600)  # Cache for 1 hour
    
    return user
\`\`\`

**Pros:**
- Only caches data that's actually requested (efficient)
- Cache failures don't break system (degrades to database)
- Simple to implement

**Cons:**
- Cache miss penalty (3 operations: check cache, query DB, write cache)
- Initial requests always miss (cold start)
- Can have stale data if not invalidated properly

**When to use:** Most general-purpose caching scenarios.

---

### **2. Read-Through Cache**

**Flow:**
1. Application requests data from cache
2. Cache library handles database query if miss
3. Cache automatically loads and returns data

**Difference from Cache-Aside:** Cache library manages database interaction (not application).

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    # Cache library handles everything
    return cache.get_or_load(f"user:{user_id}")
    
# Cache library internally:
# - Checks cache
# - If miss, calls registered loader function
# - Stores result in cache
# - Returns data
\`\`\`

**Pros:**
- Cleaner application code
- Consistent caching logic

**Cons:**
- Tighter coupling (cache must know about database)
- Less flexibility

**When to use:** When cache library supports it (e.g., Rails caching, some Java frameworks).

---

## Cache Writing Patterns

### **1. Write-Through Cache**

**Flow:**
1. Application writes data to cache
2. Cache **synchronously** writes to database
3. Return success only after database write succeeds

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write to cache, which writes through to database
    cache.set(f"user:{user_id}", new_data)  # Internally writes to DB
    return success
\`\`\`

**Pros:**
- Data consistency (cache and DB always in sync)
- Cache is always up-to-date
- No stale data

**Cons:**
- Slower writes (every write hits database)
- Write latency = cache write + database write
- Wasted cache space (might cache data never read)

**When to use:** When consistency is critical and stale data is unacceptable.

---

### **2. Write-Back (Write-Behind) Cache**

**Flow:**
1. Application writes data to cache
2. Return success immediately
3. Cache **asynchronously** writes to database later (batched)

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write to cache only
    cache.set(f"user:{user_id}", new_data)  # Instant return
    # Cache will flush to database periodically
    return success
\`\`\`

**Pros:**
- Fast writes (only to memory)
- Can batch multiple writes to database (more efficient)
- Reduces database write load

**Cons:**
- Risk of data loss (if cache crashes before flushing to database)
- Complex to implement
- Eventual consistency

**When to use:** Write-heavy workloads where some data loss is acceptable (e.g., view counts, analytics).

---

### **3. Write-Around Cache**

**Flow:**
1. Application writes directly to database
2. Cache is bypassed on write
3. Data loaded into cache only when read (cache-aside pattern)

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write directly to database
    database.update("UPDATE users SET ... WHERE id = ?", user_id, new_data)
    
    # Optionally invalidate cache
    cache.delete(f"user:{user_id}")
    
    return success
\`\`\`

**Pros:**
- Avoids cache pollution (don't cache data that won't be read)
- Simpler than write-through

**Cons:**
- Cache miss on first read after write
- Stale data possible if not invalidated

**When to use:** Write-once, read-rarely data (e.g., logs, historical data).

---

## Cache Eviction Policies

**Problem**: Cache has limited memory. When full, which items should be removed?

### **1. Least Recently Used (LRU)**

**Policy**: Evict the item that hasn't been accessed for the longest time.

**Example:**
- Cache holds: A (accessed 10 min ago), B (accessed 2 min ago), C (accessed 5 min ago)
- Cache full, need to evict
- **Evict A** (least recently used)

**Pros:**
- Good for most workloads
- Keeps hot data in cache
- Simple to understand

**Cons:**
- Doesn't account for access frequency
- Can evict items that are frequently accessed but haven't been accessed recently

**Implementation:** Doubly linked list + hash map

**This is the most common eviction policy!**

---

### **2. Least Frequently Used (LFU)**

**Policy**: Evict the item accessed the fewest times.

**Example:**
- A: 100 accesses
- B: 5 accesses
- C: 50 accesses
- **Evict B** (least frequently used)

**Pros:**
- Keeps truly hot data (frequently accessed)

**Cons:**
- Old popular items stay forever (accessed 1000× last year, 0× this year)
- Complexity

**When to use:** When access frequency more important than recency (recommendation systems).

---

### **3. First In First Out (FIFO)**

**Policy**: Evict oldest item, regardless of access patterns.

**Example:**
- Items added: A, B, C
- **Evict A** (first in)

**Pros:**
- Simple

**Cons:**
- Ignores access patterns
- Poor performance

**Rarely used in practice.**

---

### **4. Time To Live (TTL)**

**Policy**: Evict items after a specified time, regardless of access.

**Example:**
- Cache user profile for 1 hour
- After 1 hour, automatically evicted (even if accessed frequently)

**Pros:**
- Guarantees data freshness
- Simple

**Cons:**
- Might evict hot data
- Cache misses when TTL expires

**Common practice:** Combine TTL with LRU (both limits).

---

### **5. Random Replacement**

**Policy**: Evict a random item.

**Pros:**
- Simple, fast

**Cons:**
- Unpredictable
- Might evict hot data

**When to use:** When simplicity > performance, or when access patterns are truly random.

---

## Cache Invalidation

**Two hard problems in computer science: cache invalidation, naming things, and off-by-one errors.**

**Challenge**: How to ensure cache stays consistent with database?

### **Strategy 1: Time-Based (TTL)**

Set expiration time on cached data.

**Example:**
\`\`\`python
cache.set("user:123", user_data, ttl=3600)  # Expire after 1 hour
\`\`\`

**Pros:**
- Simple
- Automatic

**Cons:**
- Stale data until TTL expires
- Too short TTL = frequent cache misses
- Too long TTL = stale data

**When to use:** Data that changes infrequently, stale data acceptable.

---

### **Strategy 2: Explicit Invalidation**

Delete from cache when data changes.

**Example:**
\`\`\`python
def update_user(user_id, new_data):
    # Update database
    database.update(user_id, new_data)
    
    # Invalidate cache
    cache.delete(f"user:{user_id}")
\`\`\`

**Pros:**
- No stale data (cache always fresh)

**Cons:**
- Must invalidate in every write path
- Easy to miss a write path (leads to stale data)
- Race conditions possible

**When to use:** When data consistency is critical.

---

### **Strategy 3: Write-Through**

Update cache and database together (discussed above).

**Pros:**
- Cache always consistent

**Cons:**
- Slower writes

---

### **Strategy 4: Event-Based Invalidation**

Publish events when data changes; cache listeners invalidate.

**Example:**
1. User service updates user → publishes event: \`user.updated: 123\`
2. Cache service subscribes to events → invalidates \`user: 123\`

**Pros:**
- Decoupled (services don't need to know about cache)
- Scalable

**Cons:**
- Complexity (need event system)
- Eventual consistency

**When to use:** Large systems with many services.

---

## Cache Stampede Problem

**Problem**: Cache expires, many requests simultaneously hit database (thundering herd).

**Scenario:**
1. Cache for popular item expires
2. 10,000 concurrent requests arrive
3. All 10,000 requests miss cache
4. All 10,000 query database simultaneously
5. Database overwhelmed!

**Solution 1: Locking**

First request acquires lock, queries database, updates cache. Other requests wait for cache to be populated.

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    user = cache.get(f"user:{user_id}")
    if user:
        return user
    
    # Acquire lock
    with cache.lock(f"lock:user:{user_id}", timeout=5):
        # Double-check cache (might have been populated while waiting for lock)
        user = cache.get(f"user:{user_id}")
        if user:
            return user
        
        # Query database
        user = database.query(user_id)
        cache.set(f"user:{user_id}", user, ttl=3600)
        return user
\`\`\`

---

**Solution 2: Probabilistic Early Expiration**

Randomly refresh cache before TTL expires.

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    user, ttl_remaining = cache.get_with_ttl(f"user:{user_id}")
    
    if user and ttl_remaining > 0:
        # Probabilistically refresh early
        if random.random() < (1.0 / ttl_remaining):
            # Asynchronously refresh
            background_task.enqueue(refresh_user_cache, user_id)
        return user
    
    # Cache miss: load and cache
    user = database.query(user_id)
    cache.set(f"user:{user_id}", user, ttl=3600)
    return user
\`\`\`

---

## Cache Consistency Models

### **Strong Consistency**

Cache always reflects latest database state.

**Implementation**: Write-through cache or synchronous invalidation.

**Use case**: Financial data, inventory counts (stale data unacceptable).

---

### **Eventual Consistency**

Cache may be temporarily stale but eventually consistent.

**Implementation**: TTL or asynchronous invalidation.

**Use case**: Social media (OK if user sees profile updated 5 seconds later).

---

## Distributed Caching (Redis Example)

**Redis**: In-memory data structure store, commonly used as cache.

### **Key Features:**

**1. Data Structures:**
- Strings (most common)
- Hashes (store objects)
- Lists (queues)
- Sets (unique items)
- Sorted Sets (leaderboards)

**2. Persistence:**
- RDB snapshots (periodic saves to disk)
- AOF (append-only file, logs every write)

**3. Replication:**
- Master-slave replication
- Read replicas for scaling reads

**4. Clustering:**
- Sharding across multiple nodes
- Automatic failover

---

### **Example: Caching User Profile**

\`\`\`python
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_user(user_id):
    # Try cache first
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss: query database
    user = database.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache (expire after 1 hour)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))
    
    return user

def update_user(user_id, new_data):
    # Update database
    database.update(user_id, new_data)
    
    # Invalidate cache
    r.delete(f"user:{user_id}")
\`\`\`

---

## Interview Tips

### **Common Questions:**

**Q: "How would you reduce database load in your system?"**

✅ Good answer: "Add caching layer (Redis):
1. Cache-aside pattern for reads
2. Invalidate on writes
3. TTL of 1 hour
4. Expected 90% cache hit rate
5. Reduces database load by 90%"

**Q: "What happens if the cache goes down?"**

✅ Good answer: "Two approaches:
1. **Graceful degradation**: Application falls back to database (slower but functional)
2. **High availability**: Redis Sentinel or Redis Cluster for automatic failover"

**Q: "How do you decide what to cache?"**

✅ Good answer: "Cache data that is:
1. **Read-heavy**: Read 100×more than written
2. **Expensive to compute**: Complex queries, API calls
3. **Frequently accessed**: Hot data (80/20 rule: 20% of data accounts for 80% of requests)
4. **Tolerate staleness**: Eventual consistency OK"

---

## Key Takeaways

1. **Caching dramatically improves performance**: 2ms cache vs 50ms database
2. **Cache-aside most common**: Check cache first, query DB on miss, populate cache
3. **Invalidation is hard**: Use TTL + explicit invalidation for consistency
4. **Distributed cache (Redis) standard**: Shared across app servers, survives restarts
5. **LRU most common eviction**: Evict least recently used items when cache full
6. **Target 80-90% cache hit rate**: Monitor and optimize
7. **Cache stampede prevention**: Use locking or probabilistic early expiration
8. **High availability**: Redis Sentinel/Cluster for cache reliability`,
};
