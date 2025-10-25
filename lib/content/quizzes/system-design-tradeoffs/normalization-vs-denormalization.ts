/**
 * Quiz questions for Normalization vs Denormalization section
 */

export const normalizationvsdenormalizationQuiz = [
  {
    id: 'q1',
    question:
      'You are designing a social media timeline feature like Twitter. Should you normalize or denormalize the timeline data? Explain your approach for both write and read paths.',
    sampleAnswer:
      "Social media timeline design - Hybrid normalized/denormalized approach: WRITE PATH (Normalized): When user creates a tweet: Write to normalized Tweets table: { tweet_id, user_id, content, created_at }. Single row insert (fast). Data integrity maintained (normalized schema). DENORMALIZED READ (Fan-out on write for most users): When tweet created, asynchronously fan-out to followers' timelines. For each follower: INSERT INTO timelines (follower_id, tweet_id, created_at, author_name, author_avatar, content). This denormalizes data: tweet content + author info duplicated for each follower. Example: User with 1,000 followers → 1,000 timeline inserts. READ PATH: When user loads timeline: SELECT * FROM timelines WHERE follower_id = current_user ORDER BY created_at DESC LIMIT 50. Fast query (no JOINs, pre-computed, indexed by follower_id). Latency: 10-50ms. HYBRID FOR CELEBRITIES: For users with >1M followers, fan-out is too expensive (millions of writes per tweet). Instead, use normalized pull model: Store tweet in Tweets table only. When follower loads timeline: Merge regular timeline (denormalized) + celebrity tweets (pulled from Tweets table). CONSISTENCY STRATEGY: Denormalized timeline data may be slightly stale: If user changes name, old tweets in timelines have old name. Eventually consistent: Background job updates timelines periodically. Acceptable trade-off for performance. TRADE-OFFS: Write complexity: One tweet → thousands of timeline inserts (slow for celebrities). Read performance: Timeline loads in <50ms (excellent UX). Storage: Denormalized timelines consume more storage (acceptable for performance). RESULT: Twitter uses this hybrid approach. 99% of users get denormalized timelines (fast reads). Celebrities use pull model (avoid expensive fan-out). Best of both worlds: fast reads for most users, manageable writes for everyone.",
    keyPoints: [
      'Normalize tweets table (write path)',
      'Denormalize timelines table (fan-out on write)',
      'Hybrid: Fan-out for regular users, pull for celebrities',
      'Trade-off: Write complexity for read performance',
      'Eventual consistency acceptable for timelines',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the trade-offs between normalized and denormalized approaches for an e-commerce product catalog with millions of products and billions of pageviews per month.',
    sampleAnswer:
      'E-commerce product catalog normalization analysis: SCALE: Millions of products, billions of pageviews/month. Read:write ratio: 1000:1 (1,000 views per product update). NORMALIZED APPROACH: Schema: Products (id, name, category_id, price), Categories (id, name), Reviews (id, product_id, rating, comment). Query to display product: JOIN Products + Categories + aggregate Reviews for avg_rating and review_count. PROS: Easy to update product (single row update). Category changes propagate automatically (foreign key). Reviews automatically affect aggregations (no manual update). CONS: Complex query with JOIN + aggregation on every pageview. Slow query time: 100-500ms for complex JOIN + aggregation. Difficult to scale horizontally (JOINs across shards expensive). At billions of pageviews, this is unacceptable latency. DENORMALIZED APPROACH: Schema: Products (id, name, category_id, category_name, price, avg_rating, review_count, last_updated). Query: SELECT * FROM products WHERE category_id = 1. PROS: Simple query (no JOINs, no aggregations). Fast query time: 5-10ms (single table, indexed). Easy to cache (simple query, no joins). Horizontal scaling friendly (no cross-shard JOINs). CONS: Redundant data (category_name duplicated across products). Complex writes: When review added, must update products.avg_rating and products.review_count. Stale data: Aggregations updated asynchronously (eventual consistency). More storage (denormalized columns). IMPLEMENTATION: Products table stores denormalized avg_rating and review_count. When review added: Insert into Reviews table (normalized). Publish "review.created" event. Background worker consumes event, recalculates product avg_rating, updates Products table. Latency: Product stats updated within 1-5 seconds (eventual consistency). Users don\'t notice stale rating briefly. CACHING LAYER: Further optimize with Redis cache: Cache product details for 1 hour. Invalidate cache when product updated. Cache hit rate: >90% (most products viewed repeatedly). RESULT: Billions of reads served from cache (5-10ms latency). Writes handled by background workers (eventual consistency). Acceptable staleness: Product rating updated within 5 seconds. Storage trade-off: Denormalized columns + cache increase storage, but performance gains are worth it. For read-heavy catalog, denormalization is correct choice.',
    keyPoints: [
      'Read:write ratio 1000:1 favors denormalization',
      'Normalized: Slow queries (100-500ms) at scale',
      'Denormalized: Fast queries (5-10ms), eventual consistency',
      'Background workers update aggregations asynchronously',
      'Caching layer (Redis) further optimizes denormalized queries',
    ],
  },
  {
    id: 'q3',
    question:
      'You need to maintain data consistency in a denormalized database where user names are duplicated across multiple tables. What strategies would you use to keep the data consistent?',
    sampleAnswer:
      'Maintaining consistency in denormalized database - strategies: PROBLEM: User name stored in multiple tables: Users (normalized), Posts (denormalized), Comments (denormalized), Likes (denormalized). When user changes name, must update all tables. STRATEGY 1: DATABASE TRIGGERS (Strongest consistency). Implementation: CREATE TRIGGER update_user_name_in_posts AFTER UPDATE OF name ON users FOR EACH ROW BEGIN UPDATE posts SET author_name = NEW.name WHERE user_id = NEW.user_id; UPDATE comments SET author_name = NEW.name WHERE user_id = NEW.user_id; UPDATE likes SET user_name = NEW.name WHERE user_id = NEW.user_id; END; PROS: Automatic, synchronous (immediate consistency). Transactional (all updates succeed or fail together). No application code changes needed. CONS: Performance impact (multiple writes in single transaction). Can cause lock contention. Not scalable for high-volume updates. STRATEGY 2: APPLICATION-LEVEL TRANSACTION. Implementation (pseudo-code): function updateUserName (userId, newName) { db.beginTransaction(); try { db.users.update({ id: userId }, { name: newName }); db.posts.updateMany({ user_id: userId }, { author_name: newName }); db.comments.updateMany({ user_id: userId }, { author_name: newName }); db.likes.updateMany({ user_id: userId }, { user_name: newName }); db.commit(); } catch (error) { db.rollback(); throw error; } }. PROS: Explicit control in application code. Can add retry logic, error handling. Works across microservices (distributed transaction with Saga pattern). CONS: Slower than trigger (multiple round-trips). No automatic enforcement (developer can forget). Distributed transactions complex. STRATEGY 3: EVENTUAL CONSISTENCY (Async updates). Implementation: When user updates name: Update Users table (normalized). Publish "user.name_changed" event to message queue. Background workers consume event: Update posts WHERE user_id = userId. Update comments WHERE user_id = userId. Update likes WHERE user_id = userId. Latency: 1-10 seconds for full consistency. PROS: Fast user update (no wait for denormalized updates). Scalable (async workers can scale independently). Resilient (retries on failure). CONS: Temporary inconsistency (old name visible for seconds). Eventual consistency complexity. STRATEGY 4: VERSIONED CACHE (Avoid denormalization). Instead of denormalizing user name, store user_id only. Cache user data in Redis with version: redis.set("user:101", JSON.stringify({ name: "Alice", version: 5 })). When displaying post, fetch user from cache (fast). When user changes name, increment version and invalidate cache. PROS: No denormalized data (single source of truth). Fast reads (cache hit). Easy consistency (update one place). CONS: Requires cache (additional infrastructure). Cache misses require DB query. RECOMMENDATION FOR SOCIAL MEDIA: Use EVENTUAL CONSISTENCY (Strategy 3): User name changes are rare (< 0.01% of requests). Fast user experience critical (don\'t wait for denormalized updates). Brief inconsistency acceptable (old posts show old name for few seconds). Scalable (millions of posts updated async). Trade-off: Slight temporary inconsistency for performance and scale.',
    keyPoints: [
      'Database triggers: Synchronous, strong consistency, but slow',
      'Application transactions: Explicit control, works across services',
      'Eventual consistency: Async updates, fast, scalable, brief staleness',
      'Versioned cache: Avoid denormalization entirely, use cache',
      'Choose based on consistency requirements vs performance needs',
    ],
  },
];
