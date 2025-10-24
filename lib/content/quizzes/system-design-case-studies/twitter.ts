/**
 * Quiz questions for Design Twitter section
 */

export const twitterQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fanout-on-write vs fanout-on-read approaches for Twitter timeline generation. Why does Twitter use a hybrid approach, and what is the "celebrity problem"?',
    sampleAnswer:
      "FANOUT-ON-WRITE (Push): When user Bob posts tweet, immediately push it to timelines of ALL his followers. PROS: Fast reads (pre-computed), scales reads to millions/sec. CONS: Slow writes (if Bob has 10K followers, must write 10K times), storage explosion (tweet duplicated across followers' timelines). FANOUT-ON-READ (Pull): When user Alice requests timeline, query tweets from all her followings and merge. PROS: Fast writes (single insert), minimal storage. CONS: Slow reads (heavy joins), cannot scale to millions of timeline requests. CELEBRITY PROBLEM: If celebrity with 100M followers posts tweet, fanout-on-write would require 100M writes (minutes to complete, impossible to scale). HYBRID SOLUTION: (1) Regular users (<10K followers): Use fanout-on-write. 99% of users have <10K followers, so this works for majority. (2) Celebrities (>=10K followers): Use fanout-on-read. When generating timeline, separately query recent tweets from celebrities user follows and merge with pre-computed timeline. (3) Caching: Cache generated timelines in Redis (5-min TTL) to absorb read load. RESULT: Fast writes for normal users, manageable reads for all users, no celebrity problem. REAL-WORLD: Twitter uses fanout-on-write for <10K threshold, Instagram uses different threshold based on engagement rates. KEY INSIGHT: No single approach works at scale - hybrid is production reality.",
    keyPoints: [
      'Fanout-on-write: Fast reads, slow writes, storage explosion',
      'Fanout-on-read: Fast writes, slow reads, complex queries',
      'Celebrity problem: 100M followers × fanout-on-write = impossible',
      'Hybrid: <10K followers fanout-on-write, >=10K fanout-on-read',
      'Redis caching with 5-min TTL critical for read scalability',
    ],
  },
  {
    id: 'q2',
    question:
      'Design the database sharding strategy for Twitter. What tables do you shard, what is the shard key, and how do you handle queries like "Get followers of celebrity X"?',
    sampleAnswer:
      'SHARDING STRATEGY: (1) TWEETS TABLE - Shard by user_id. Rationale: User timeline query "SELECT * FROM tweets WHERE user_id = X ORDER BY created_at DESC" hits single shard (efficient). Formula: shard = user_id % NUM_SHARDS. All tweets from same user stay together. (2) FOLLOWS TABLE - Shard by follower_id. Rationale: Query "Get who Alice follows" (SELECT followee_id FROM follows WHERE follower_id = Alice) hits single shard. This is the common query for timeline generation. (3) USERS TABLE - Shard by user_id. PROBLEM QUERY: "Get followers of celebrity X" requires reverse query: SELECT follower_id FROM follows WHERE followee_id = X. Since sharded by follower_id, this requires scatter-gather across ALL shards (slow). SOLUTIONS: (1) Accept scatter-gather for rare query (showing follower list is uncommon vs timeline generation). (2) Maintain denormalized reverse index: Create separate table followers_of sharded by followee_id. Write to both tables on follow/unfollow. (3) For celebrities, cache follower list in Redis (updated hourly). PRODUCTION CHOICE: Scatter-gather acceptable because "show followers" is rare (1% of queries) compared to timeline generation (99% of queries). Optimize for common case. KEY INSIGHT: Sharding forces you to choose which queries are fast - cannot optimize all queries in sharded system. Choose shard key based on most frequent query pattern.',
    keyPoints: [
      'Shard tweets by user_id (user timeline query hits single shard)',
      'Shard follows by follower_id (timeline generation efficient)',
      'Reverse queries (get followers) require scatter-gather',
      'Optimize for common case (timeline) not rare case (follower list)',
      'Denormalized reverse index possible but adds write complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'Twitter generates 10 billion timeline views per day. Walk through your caching strategy: what to cache, cache invalidation, and how you handle cache failures.',
    sampleAnswer:
      "CACHING STRATEGY: (1) WHAT TO CACHE: Cache generated timelines in Redis. Key: timeline:{user_id}, Value: [tweet_ids]. Cache top 100 tweets per user (covers 99% of timeline views since users rarely scroll past page 3). (2) CACHE SIZE: 10B views/day, assume 20M daily active users (users view timeline multiple times). Cache timelines for top 5M most active users (those who refresh often). 5M users × 100 tweets × 8 bytes (tweet_id) = 4 GB timeline IDs. Add hydrated tweet data (300 bytes per tweet): 5M × 100 × 300 = 150 GB. Total: ~750 GB Redis cache. (3) TTL: 5 minutes. Timelines go stale quickly as new tweets arrive, but 5-min staleness acceptable for social media. Reduces cache refresh load while keeping content fresh. (4) CACHE INVALIDATION: Passive expiry (TTL), no active invalidation. Why? Active invalidation on every tweet post would require O(followers) cache deletes (worse than fanout problem). Let TTL handle it. (5) CACHE WARMING: On cache miss, compute timeline (hybrid fanout) and store in Redis. High cache hit rate (80%+) expected due to user viewing patterns (morning check, lunch check, evening check). (6) CACHE FAILURE: If Redis cluster goes down: (a) Circuit breaker pattern - detect Redis down, bypass cache. (b) Serve timelines directly from database (degraded performance). (c) Rate limit timeline requests to protect DB (e.g., allow 10K QPS instead of 115K). (d) Fallback: Serve cached timelines from application-level cache (limited, but helps). (7) REDIS ARCHITECTURE: Use Redis Cluster with 3-5 replicas per shard. Automatic failover if primary fails. RESULT: 80% cache hit rate → Only 23K timeline queries hit DB (vs 115K) - manageable with read replicas. KEY INSIGHT: Cache is not optional at this scale - it's a critical dependency. Plan for cache failures with circuit breakers and degraded service mode.",
    keyPoints: [
      "Cache 5M active users' timelines = 750 GB Redis",
      'TTL: 5 minutes (passive expiry, no active invalidation)',
      'Cache hit rate: 80%+ reduces DB load from 115K to 23K QPS',
      'Cache failure: Circuit breaker + rate limiting + degraded mode',
      'Redis Cluster with replication for high availability',
    ],
  },
];
