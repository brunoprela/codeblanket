/**
 * Instagram Architecture Section
 */

export const instagramarchitectureSection = {
   id: 'instagram-architecture',
   title: 'Instagram Architecture',
   content: `Instagram is one of the world's largest social media platforms with over 2 billion users, sharing 100+ million photos and videos daily. Acquired by Facebook (now Meta) in 2012 for $1 billion, Instagram has evolved from a simple photo-sharing app to a comprehensive social platform. This section explores the architecture that enables Instagram to scale globally while maintaining performance.

## Overview

Instagram\'s architecture is notable for several achievements:
- **2+ billion users** with 500 million daily active users
- **100+ million photos/videos** uploaded daily
- **4.2 billion likes** per day
- **Started with Django monolith**, evolved to microservices for scale
- **Heavy use of caching** with Memcached and Redis
- **PostgreSQL** for relational data, **Cassandra** for feeds

### Key Architectural Principles

1. **Keep it simple**: Start with proven technologies (Django, PostgreSQL)
2. **Scale incrementally**: Don't over-engineer early
3. **Measure everything**: Data-driven decisions
4. **Optimize hot paths**: Focus on critical user flows (feed, upload, like)
5. **Embrace caching**: Aggressive caching at all layers

---

## Evolution: Monolith to Hybrid Architecture

### Phase 1: Monolithic Django App (2010-2012)

Initially, Instagram was a simple Django monolith:

\`\`\`
User Request → Load Balancer → Django App Servers
                                      ↓
                               PostgreSQL Database
\`\`\`

**Technology Stack**:
- **Django**: Python web framework
- **PostgreSQL**: Primary data store
- **Gunicorn**: Python WSGI HTTP server
- **NGINX**: Reverse proxy and load balancer
- **S3**: Photo storage
- **CloudFront**: CDN for photo delivery

**Why Django?**:
- Rapid development (launched MVP in 8 weeks)
- Strong community and ecosystem
- Admin interface out of the box
- ORM for database access

**Scaling Challenges**:
- Database bottleneck (reads and writes hitting single instance)
- Monolith becoming hard to change
- Need for independent scaling of components

---

### Phase 2: Database Sharding (2012-2014)

As user base grew, single PostgreSQL instance couldn't handle load.

**Approach: Horizontal Sharding**

\`\`\`
User Request → Django → Shard Router
                            ↓
        Shard 1 (user 1-1M) | Shard 2 (user 1M-2M) | Shard 3 (user 2M-3M)
\`\`\`

**Sharding Strategy**:
- **Partition by user_id**: Hash user ID to determine shard
- Each shard is a PostgreSQL instance
- Shard count: Started with 8, scaled to hundreds

**Challenges**:
- Cross-shard queries (following relationships across shards)
- Rebalancing when adding shards
- Foreign key constraints don't work across shards

**Solutions**:
- Denormalize data to avoid cross-shard queries
- Application-level joins when necessary
- Use consistent hashing for minimal data movement

---

### Phase 3: Microservices for Core Features (2015-present)

Extracted core features into microservices:

**Services**:
- **Feed Service**: Generate user feeds
- **Media Service**: Photo/video upload and processing
- **User Service**: User profiles and relationships
- **Direct Messaging Service**: Instagram DMs
- **Stories Service**: Stories and ephemeral content
- **Search Service**: User and hashtag search

**Why Microservices?**:
- **Independent scaling**: Feed service needs more capacity than others
- **Technology flexibility**: Use Python, Java, Go based on needs
- **Team autonomy**: Small teams own services end-to-end
- **Fault isolation**: One service failure doesn't bring down entire system

**Communication**:
- **Synchronous**: gRPC for low latency, request-response
- **Asynchronous**: Kafka for event streaming, eventual consistency

---

## Core Architecture Components

### 1. Photo Storage and Delivery

Instagram stores and delivers billions of photos efficiently.

**Upload Flow**:

1. **Client uploads** photo to Instagram server
2. **Resize and optimize**: Generate multiple sizes (thumbnail, medium, full)
3. **Store in Amazon S3**: Master and resized versions
4. **Generate unique URL**: Return to client
5. **Update metadata**: Store in database (photo_id, user_id, timestamp, caption, filters)

**Storage Optimization**:
- **Compression**: Aggressive JPEG compression (reduce file size by 50%+)
- **Progressive rendering**: Store photos in progressive JPEG (loads blurry→clear)
- **Lazy loading**: Load images as user scrolls (don't load entire feed upfront)

**Delivery via CDN (Facebook CDN)**:
- Photos served from edge locations worldwide
- Cache hit rate: 95%+
- User requests photo → CDN → Cache hit (fast) or fetch from S3 (slower)

**Image Processing**:
- **Filters**: Apply filters client-side when possible (reduce server load)
- **Face detection**: Identify faces for tagging (ML models)
- **Content moderation**: Detect inappropriate content (ML classifiers)

**Scale**:
- 100 million photos/videos uploaded daily
- Petabytes of storage
- Billions of image requests per day

---

### 2. Feed Generation

Instagram\'s feed is a personalized timeline of photos/videos from followed accounts.

**Challenge**: User with 1,000 followers, each posting 10 times/day = 10,000 potential posts. How to select and rank?

**Feed Architecture**:

**Fanout-on-Write vs Fanout-on-Read**:

**Fanout-on-Write (Used for celebrities)**:
\`\`\`
When celebrity posts → Write to all followers' feeds immediately
Pros: Fast read (feed pre-computed)
Cons: Slow write (write to millions of feeds)
\`\`\`

**Fanout-on-Read (Used for regular users)**:
\`\`\`
When user requests feed → Fetch recent posts from followed accounts → Rank
Pros: Fast write (no fanout)
Cons: Slow read (compute at read time)
\`\`\`

**Instagram's Hybrid Approach**:
- Regular users: Fanout-on-write (most followers have <1,000 followers)
- Celebrities: Fanout-on-read (avoid writing to 100M+ feeds)
- Threshold: ~1 million followers

**Feed Generation Steps**:

1. **Fetch Candidate Posts**:
   - Get posts from followed accounts (last 7 days)
   - Query Cassandra (feed storage)

2. **Ranking**:
   - ML model predicts engagement probability
   - Features: User preferences, post content, relationship strength, timeliness
   - Score each post

3. **Filter**:
   - Remove already seen posts
   - Remove hidden/muted accounts
   - Apply content policies

4. **Pagination**:
   - Return top N posts (e.g., 50)
   - Infinite scroll loads more

**Caching**:
- Feed cached in Redis for ~5 minutes
- Subsequent requests within 5 min → Cache hit (fast)
- Background job refreshes cache periodically

**Performance**:
- P50 feed load: <200ms
- P99 feed load: <500ms

---

### 3. Relationships and Graph Storage

Instagram\'s core is a social graph: users follow other users.

**Data Model** (PostgreSQL):

\`\`\`sql
Table: users
- user_id (primary key)
- username
- bio
- profile_pic_url

Table: follows
- follower_id (who is following)
- followee_id (who is being followed)
- created_at
- PRIMARY KEY (follower_id, followee_id)
- INDEX on followee_id (to get followers)
\`\`\`

**Sharding**:
- Users table sharded by user_id
- Follows table sharded by follower_id
- Challenge: Getting follower count requires cross-shard query

**Denormalization for Performance**:
- Store follower_count and following_count on user record
- Updated asynchronously via background jobs
- Accept slight inconsistency (eventual consistency)

**Challenges**:

**1. Followers Query**:
- Problem: User with 100M followers → Query returns 100M rows
- Solution: Pagination (return first 1,000, then load more)

**2. Mutual Followers**:
- Problem: Show "X and Y follow this user" (set intersection across shards)
- Solution: Pre-compute for celebrities, sample for others

**3. Follow Recommendation**:
- Problem: Suggest users to follow (complex graph algorithm)
- Solution: Offline batch jobs compute suggestions, store in cache

---

### 4. Like and Comment System

Likes and comments are high-volume, real-time interactions.

**Data Model** (Cassandra):

\`\`\`
Table: likes
Partition Key: photo_id
Clustering Key: user_id
Columns: timestamp

Query: "Get all likes for photo X" → Single partition read
Query: "Did user Y like photo X?" → Single row read
\`\`\`

**Why Cassandra?**:
- High write throughput (millions of likes/second)
- Scalable (add nodes for capacity)
- Denormalized data model (optimize for reads)

**Like Counter**:
- Problem: Count likes for photo (requires scanning partition)
- Solution: Separate counter table updated asynchronously

\`\`\`
Table: like_counts
Partition Key: photo_id
Columns: count

Update: When user likes photo → Increment counter (eventual consistency)
\`\`\`

**Comments**:

Similar to likes but with additional content:

\`\`\`
Table: comments
Partition Key: photo_id
Clustering Key: timestamp DESC
Columns: user_id, text, mentions

Query: "Get recent comments for photo X" → Single partition, time-ordered
\`\`\`

**Real-Time Updates**:
- User likes photo → Increment like count in UI immediately (optimistic update)
- Backend processes like asynchronously → Eventually consistent
- WebSocket connection pushes updates to other clients viewing the same photo

---

### 5. Stories

Instagram Stories (24-hour ephemeral content) have different requirements than permanent posts.

**Characteristics**:
- **Ephemeral**: Auto-delete after 24 hours
- **High volume**: 500 million users post stories daily
- **Sequential viewing**: Users watch stories in order (not ranked feed)

**Architecture**:

**Storage**:
- Videos/photos stored in S3 with 24-hour TTL
- Metadata in Redis (expire after 24 hours automatically)

**Data Model** (Redis):

\`\`\`
Key: stories:user:123
Value: [story1_id, story2_id, story3_id]
TTL: 24 hours

Key: story:story1_id
Value: {user_id: 123, media_url: "s3://...", timestamp: ...}
TTL: 24 hours
\`\`\`

**Why Redis?**:
- Auto-expiration (no manual cleanup needed)
- Fast reads (in-memory)
- List data structure for ordered stories

**Stories Feed**:

1. User opens stories
2. Fetch followed users' stories (Redis: MGET stories:user:*)
3. Order by timestamp (most recent first)
4. Pre-fetch first few videos (predictive loading)

**Viewed Tracking**:
- Track who viewed each story (for creator analytics)
- Store in Cassandra (high write volume)
- Aggregate counts asynchronously

---

### 6. Direct Messaging (Instagram DMs)

Instagram DMs is a real-time messaging system.

**Requirements**:
- Low latency (<100ms)
- Real-time delivery (WebSocket)
- Message history (persistent storage)
- Read receipts, typing indicators

**Architecture**:

**Messaging Protocol**:
- **WebSocket** for persistent connection
- Client maintains connection to messaging server
- Server pushes messages instantly

**Message Storage** (Cassandra):

\`\`\`
Table: messages
Partition Key: conversation_id
Clustering Key: timestamp DESC
Columns: sender_id, text, media_url, message_id
\`\`\`

**Message Delivery Flow**:

1. User A sends message to User B
2. Client → WebSocket → Messaging Server
3. Server writes message to Cassandra
4. Server looks up User B's connected server (via Redis)
5. Server pushes message to User B's WebSocket connection
6. User B receives message instantly

**Presence and Typing Indicators**:
- Stored in Redis (ephemeral data)
- TTL of 60 seconds (if no heartbeat, user considered offline)

\`\`\`
Key: presence:user:123
Value: {status: "online", last_seen: timestamp}
TTL: 60 seconds
\`\`\`

**Offline Messages**:
- If User B offline, message stored in Cassandra
- When User B comes online, fetch undelivered messages
- Push notification sent to User B's device

---

### 7. Search

Instagram search covers users, hashtags, and locations.

**Search Types**:

**1. User Search**:
- Search by username or full name
- Example: Search "john" → ["john_doe", "johnny", "john.smith"]

**2. Hashtag Search**:
- Search for hashtags (#travel, #food)
- Show post count, related hashtags

**3. Location Search**:
- Search places (New York, Eiffel Tower)
- Show posts tagged with location

**Architecture**:

**Elasticsearch** for search:
- Index users, hashtags, locations
- Full-text search with typo tolerance
- Ranking based on popularity, recency

**User Index**:
\`\`\`json
{
  "user_id": 123,
  "username": "john_doe",
  "full_name": "John Doe",
  "follower_count": 5000,
  "is_verified": true
}
\`\`\`

**Ranking Factors**:
- Follower count (popular users ranked higher)
- Verification status
- User\'s social graph (prioritize followed users)
- Engagement rate

**Autocomplete**:
- As user types, suggest completions
- Use prefix query in Elasticsearch
- Cache popular searches in Redis

**Performance**:
- P50 search: <50ms
- P99 search: <200ms

---

## Caching Strategy

Instagram uses aggressive caching at multiple layers.

### 1. CDN (Content Delivery Network)

- **Photos and videos** served from Facebook CDN
- Edge locations worldwide
- Cache hit rate: 95%+

### 2. Memcached

- **Application-level caching** for database query results
- Used for user profiles, follower lists, feed data

**Example**:
\`\`\`python
def get_user_profile (user_id):
    # Try cache first
    cache_key = f"user_profile:{user_id}"
    profile = memcached.get (cache_key)
    
    if profile:
        return profile  # Cache hit
    
    # Cache miss - query database
    profile = db.query("SELECT * FROM users WHERE user_id = ?", user_id)
    
    # Store in cache (TTL: 5 minutes)
    memcached.set (cache_key, profile, ttl=300)
    
    return profile
\`\`\`

**Cache Invalidation**:
- Time-based (TTL: 1-10 minutes for most data)
- Event-based (user updates profile → invalidate cache)
- Trade-off: Stale data vs load on database

### 3. Redis

- **Real-time data** (stories, presence, typing indicators)
- **Feed cache** (personalized feeds)
- **Rate limiting** (API rate limits per user)

**Feed Cache Example**:
\`\`\`
Key: feed:user:123
Value: [post1_id, post2_id, ..., post50_id]
TTL: 5 minutes
\`\`\`

**Benefits of Caching**:
- Reduced database load (90%+ requests served from cache)
- Lower latency (cache: <1ms, database: 10-50ms)
- Improved user experience

---

## Database Strategy

### PostgreSQL (Relational Data)

Used for structured data with relationships:
- Users
- Follows (relationships)
- Media metadata (photos/videos)

**Optimizations**:

**1. Read Replicas**:
- Primary handles writes
- Replicas handle reads (95% of traffic)
- Replication lag: <100ms

**2. Connection Pooling**:
- PgBouncer for connection pooling
- Reuse connections (reduce overhead)
- Thousands of connections pooled to hundreds

**3. Query Optimization**:
- Indexes on frequently queried columns
- Denormalize for hot queries (e.g., follower_count on user)
- Monitor slow queries (>100ms)

### Cassandra (High-Volume Data)

Used for high-throughput, denormalized data:
- Likes
- Comments
- Feed data
- Messages

**Why Cassandra?**:
- **Write optimized**: Log-structured storage (fast writes)
- **Scalable**: Linear scalability (add nodes for capacity)
- **No single point of failure**: Peer-to-peer architecture

**Data Model**:
- Denormalized (duplicate data to avoid joins)
- Partition key chosen for query pattern
- Clustering key for sorting

**Example: Likes**:
\`\`\`
# Query: Get all likes for photo X
Partition Key: photo_id (all likes for a photo in same partition)
Clustering Key: user_id

# Query: Did user Y like photo X?
Partition Key: photo_id
Clustering Key: user_id
Result: Single partition, single row lookup (very fast)
\`\`\`

**Consistency**:
- Quorum reads/writes (W=2, R=2, N=3)
- Accept eventual consistency for non-critical data

---

## Scaling Challenges and Solutions

### Challenge 1: Feed Latency

**Problem**: Generating personalized feed for user with 1,000 follows is slow.

**Solutions**:
1. **Pre-compute feeds**: Background jobs generate feeds ahead of time
2. **Cache aggressively**: Cache feed for 5 minutes in Redis
3. **Pagination**: Return top 50 posts, load more on scroll
4. **Parallel fetching**: Fetch from multiple shards in parallel

### Challenge 2: Celebrity Users

**Problem**: User with 100M followers → Fanout-on-write infeasible (100M writes per post).

**Solutions**:
1. **Fanout-on-read**: Don't pre-compute celebrity posts in followers' feeds
2. **Fetch on-demand**: When user requests feed, check if they follow celebrities, fetch recent posts
3. **Merge**: Combine regular feed (pre-computed) + celebrity posts (on-demand)

### Challenge 3: Database Hotspots

**Problem**: Popular post gets millions of likes → Single partition in Cassandra becomes hot.

**Solutions**:
1. **Partition splitting**: Split hot partitions across multiple nodes
2. **Caching**: Cache like counts in Redis (reduce Cassandra reads)
3. **Rate limiting**: Limit write rate per partition

### Challenge 4: Cross-Shard Queries

**Problem**: Show mutual followers (requires querying multiple shards).

**Solutions**:
1. **Denormalization**: Store mutual followers on user record (eventual consistency)
2. **Background jobs**: Compute offline, update periodically
3. **Sampling**: For non-critical features, sample instead of exact count

---

## Migration to TAO (Facebook\'s Data Store)

After acquisition, Instagram migrated from PostgreSQL to **TAO** for social graph data.

**TAO (The Associations and Objects)**:
- Facebook's distributed data store
- Optimized for social graph (users, relationships, content)
- Built on MySQL (sharded) + Memcached (caching)

**Benefits**:
- Leverage Facebook's infrastructure
- Better performance for graph queries
- Unified platform with Facebook

**Data Model**:
- **Objects**: Users, photos, comments
- **Associations**: Follows, likes, tags

**Caching Layer**:
- Write-through cache (writes go to cache + database)
- Read cache hit rate: 99%+

**Migration**:
- Gradual migration over 18 months
- Dual writes (both PostgreSQL and TAO during transition)
- Read from TAO, validate against PostgreSQL
- Fully migrated by 2014

---

## Observability and Monitoring

Instagram monitors system health extensively.

### Metrics

**Key Metrics**:
- **Request rate**: Requests per second per service
- **Error rate**: 5xx errors, timeouts
- **Latency**: P50, P95, P99 response times
- **Saturation**: CPU, memory, disk, network utilization

**Tools**:
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **ODS (Operational Data Store)**: Facebook\'s internal metrics system

### Alerting

**Alert Thresholds**:
- Error rate >1% → Page on-call
- P99 latency >500ms → Warning
- Feed generation >1 second → Critical

**On-Call Rotation**:
- Engineers rotate weekly
- Escalation to senior engineers for critical issues

### Incident Response

**Process**:
1. Alert fired → On-call paged
2. Investigate (logs, metrics, traces)
3. Mitigate (rollback, scale up, disable feature)
4. Post-mortem (blameless, action items)

---

## Key Lessons

### 1. Start Simple, Scale Incrementally

Instagram started with Django monolith, scaled with sharding, eventually moved to microservices. Don't over-engineer early.

### 2. Denormalize for Performance

Duplicate data to avoid expensive joins and cross-shard queries. Trade storage for speed.

### 3. Cache Aggressively

Cache at all layers (CDN, Memcached, Redis). Accept eventual consistency for non-critical data.

### 4. Measure and Optimize Hot Paths

Focus optimization efforts on critical user flows: feed load, photo upload, like action. Monitor relentlessly.

### 5. Embrace Eventual Consistency

For features like follower count, like count, accept slight delays. Perfect consistency not required.

### 6. Leverage Existing Infrastructure

After Facebook acquisition, migrated to Facebook's TAO and infrastructure. Don't reinvent the wheel.

---

## Interview Tips

**Q: How would you design Instagram\'s feed generation?**

A: Use hybrid fanout approach. For regular users (<1M followers), fanout-on-write: when user posts, write to followers' feeds immediately (stored in Cassandra). For celebrities (>1M followers), fanout-on-read: fetch recent posts when user requests feed. Merge both sources. Cache generated feed in Redis (5 min TTL). Use ML ranking model to score posts based on predicted engagement. Paginate results (return 50 posts initially, load more on scroll). Handle staleness with background refresh jobs.

**Q: How does Instagram handle millions of photo uploads per day?**

A: Photos uploaded to application servers → Resize to multiple sizes (thumbnail, medium, full) → Compress with aggressive JPEG encoding → Store in S3 → Generate unique URL → Store metadata in database (photo_id, user_id, timestamp). Serve photos via CDN (cache hit rate 95%+). Use lazy loading (load images as user scrolls). Apply filters client-side when possible to reduce server load. Asynchronous processing for face detection and content moderation (ML models).

**Q: What are the trade-offs between Cassandra and PostgreSQL for Instagram's data?**

A: PostgreSQL for structured, relational data (users, follows) where consistency and complex queries matter. Benefits: ACID transactions, foreign keys, complex joins. Drawbacks: Harder to scale writes. Cassandra for high-volume, denormalized data (likes, comments, feeds) where availability and write throughput matter. Benefits: Linear scalability, high write throughput, tunable consistency. Drawbacks: No joins (must denormalize), eventual consistency. Instagram uses both: PostgreSQL (now TAO) for social graph, Cassandra for activity streams.

---

## Summary

Instagram\'s architecture demonstrates scaling a social platform from startup to billions of users:

**Key Takeaways**:

1. **Started simple**: Django monolith with PostgreSQL
2. **Scaled incrementally**: Sharding, microservices, TAO migration
3. **Aggressive caching**: CDN, Memcached, Redis at every layer
4. **Hybrid feed approach**: Fanout-on-write for regular users, fanout-on-read for celebrities
5. **Polyglot persistence**: PostgreSQL for graph, Cassandra for activity streams
6. **Denormalization**: Duplicate data to avoid expensive queries
7. **Leverage existing infrastructure**: Migrated to Facebook TAO after acquisition
8. **Measure and optimize**: Focus on hot paths (feed, upload, like)

Instagram's evolution shows the importance of starting simple and scaling based on actual needs rather than premature optimization.
`,
};
