/**
 * Design News Feed Section
 */

export const newsFeedSection = {
    id: 'news-feed',
    title: 'Design News Feed',
    content: `A news feed (Facebook, Twitter, Instagram) shows personalized posts from friends/followees. The challenges are: generating feeds for billions of users, ranking posts by relevance, handling millions of new posts per day, real-time updates, and keeping feed generation fast (< 500ms) while maintaining consistency.

## Problem Statement

Design a news feed system that:
- **Generate Feed**: Show posts from friends/followees
- **Real-Time Updates**: New posts appear immediately
- **Ranking**: Most relevant posts first (engagement, recency)
- **Scalability**: Billions of users, millions of posts/day
- **Personalization**: User preferences, interactions
- **Media Support**: Text, images, videos in feed
- **Pagination**: Load more posts as user scrolls
- **Notifications**: Alert when friends post

**Scale**: 2 billion users, 500 million DAU, 100 million posts/day, 10K QPS for feed generation

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Post Creation**: User publishes post (text, image, video)
2. **Feed Generation**: Show posts from friends (chronological or ranked)
3. **Feed Retrieval**: User opens app, sees latest posts
4. **Interactions**: Like, comment, share posts
5. **Notifications**: Real-time alerts for new posts from close friends
6. **Privacy**: Public posts vs friends-only

### Non-Functional Requirements

1. **Low Latency**: Feed loads in < 500ms
2. **High Availability**: 99.99% uptime
3. **Scalability**: Handle spikes (New Year's Eve posts)
4. **Consistency**: Eventual consistency acceptable (feed may lag by seconds)
5. **Fault Tolerance**: Graceful degradation if services fail

---

## Step 2: Capacity Estimation

**Users**: 2 billion total, 500 million DAU

**Posts**: 100 million posts/day

**Feed Requests**: 500M users × 10 feed loads/day = 5 billion requests/day = 58K QPS average, 120K QPS peak

**Storage**:
- Posts: 100M posts/day × 1 KB (text + metadata) = 100 GB/day
- Images: 50M images/day × 500 KB = 25 TB/day
- Videos: 10M videos/day × 50 MB = 500 TB/day
- Total: ~525 TB/day = 16 PB/month

**Fanout**: Average user has 200 friends. Post creation triggers 200 feed updates (fanout).

---

## Step 3: Core Challenge - Feed Generation

**Two Approaches**:

### Approach 1: Pull Model (Read-Heavy)

**Flow**:
\`\`\`
1. User opens app
2. Query: "Get latest posts from my 200 friends"
3. Fetch posts from each friend's timeline
4. Merge 200 timelines, sort by timestamp
5. Return top 50 posts
\`\`\`

**Implementation**:

\`\`\`sql
-- Get user's friends
SELECT friend_id FROM friendships WHERE user_id = 123;
-- Returns: [456, 789, 101, ..., 200 friends]

-- Get posts from each friend (last 7 days)
SELECT * FROM posts 
WHERE user_id IN (456, 789, 101, ...) 
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC
LIMIT 50;
\`\`\`

**Pros**:
- ✅ Simple to implement
- ✅ No pre-computation (lazy, on-demand)
- ✅ Always fresh data

**Cons**:
- ❌ Slow (query 200 friends every time user opens app)
- ❌ Database load (58K QPS × 200 queries = 11.6M QPS)
- ❌ Latency scales with friend count (celebrities with 10K friends = disaster)

---

### Approach 2: Push Model (Write-Heavy, Fanout-on-Write)

**Flow**:
\`\`\`
1. User A posts "Hello World"
2. Find User A's followers: [B, C, D, ..., 200 followers]
3. Write post to each follower's feed:
   - feed:userB → [postA, ...]
   - feed:userC → [postA, ...]
   - ...
4. When User B opens app, read from feed:userB (pre-computed)
\`\`\`

**Implementation**:

\`\`\`python
# User A creates post
def create_post(user_id, content):
    # Store post
    post_id = db.insert("posts", {user_id: user_id, content: content, timestamp: now()})
    
    # Get followers
    followers = db.query("SELECT follower_id FROM followers WHERE user_id = ?", user_id)
    
    # Fanout: Write to each follower's feed
    for follower_id in followers:
        redis.lpush(f"feed:{follower_id}", post_id)  # Prepend to feed
        redis.ltrim(f"feed:{follower_id}", 0, 999)  # Keep only 1000 posts
    
    return post_id

# User B retrieves feed
def get_feed(user_id):
    post_ids = redis.lrange(f"feed:{user_id}", 0, 49)  # Top 50 posts
    posts = db.query("SELECT * FROM posts WHERE post_id IN (?)", post_ids)
    return posts
\`\`\`

**Pros**:
- ✅ Fast read (pre-computed feed in Redis)
- ✅ O(1) feed retrieval (no friend queries)
- ✅ Scales to millions of feed requests

**Cons**:
- ❌ Slow write (fanout to 200 followers takes time)
- ❌ Celebrity problem: User with 10M followers → 10M writes per post (infeasible)
- ❌ Storage: 500M users × 1000 posts × 8 bytes (post ID) = 4 TB for feed cache

---

### Approach 3: Hybrid (Best for Production)

**Strategy**: Use push for normal users, pull for celebrities.

\`\`\`
- User A (200 followers): Push model (fanout-on-write)
- User B (10M followers, celebrity): Pull model (fanout-on-read)
- User C follows: 50 normal users + 10 celebrities
  → Feed = Push (50 users' posts) + Pull (10 celebrities' posts at read time)
\`\`\`

**Implementation**:

\`\`\`python
CELEBRITY_THRESHOLD = 10000  # Users with > 10K followers

def create_post(user_id, content):
    post_id = db.insert("posts", {user_id: user_id, content: content})
    
    follower_count = db.query("SELECT COUNT(*) FROM followers WHERE user_id = ?", user_id)
    
    if follower_count < CELEBRITY_THRESHOLD:
        # Push model: Fanout to all followers
        followers = db.query("SELECT follower_id FROM followers WHERE user_id = ?", user_id)
        for follower_id in followers:
            redis.lpush(f"feed:{follower_id}", post_id)
    else:
        # Pull model: Don't fanout (too expensive)
        # Mark user as celebrity, followers will pull at read time
        redis.sadd("celebrities", user_id)

def get_feed(user_id):
    # Get pre-computed feed (push model)
    feed_posts = redis.lrange(f"feed:{user_id}", 0, 49)
    
    # Get celebrity posts (pull model)
    celebrities_following = redis.smembers(f"following_celebrities:{user_id}")
    celebrity_posts = db.query("""
        SELECT * FROM posts 
        WHERE user_id IN (?) AND created_at > NOW() - INTERVAL '7 days'
        ORDER BY created_at DESC LIMIT 20
    """, celebrities_following)
    
    # Merge and rank
    all_posts = merge_and_rank(feed_posts, celebrity_posts)
    return all_posts[:50]
\`\`\`

---

## Step 4: Ranking Algorithm

**Chronological** (Twitter, early Facebook):
- Sort by timestamp
- Simple, predictable
- But: Old posts from close friends buried by recent spam

**Engagement-Based** (Modern Facebook, Instagram):

\`\`\`python
def calculate_post_score(post, user):
    base_score = 1.0
    
    # Recency (exponential decay)
    hours_old = (now - post.created_at).hours
    recency_factor = math.exp(-0.05 * hours_old)  # 50% decay in ~14 hours
    
    # Engagement (likes, comments, shares)
    engagement_score = (post.likes * 1 + post.comments * 2 + post.shares * 3) / 10
    
    # Affinity (how close is user to post creator?)
    affinity = get_affinity(user.id, post.user_id)  # 0-1 scale
    
    # Content type
    if post.has_video:
        content_boost = 1.5  # Videos prioritized
    elif post.has_image:
        content_boost = 1.2
    else:
        content_boost = 1.0  # Text only
    
    # User preferences (learned via ML)
    user_preference = predict_engagement(user, post)  # ML model: 0-1
    
    score = base_score * recency_factor * (1 + engagement_score) * affinity * content_boost * user_preference
    return score

def get_feed_ranked(user_id):
    posts = get_feed(user_id)  # Get candidate posts
    scored_posts = [(post, calculate_post_score(post, user)) for post in posts]
    scored_posts.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    return [post for post, score in scored_posts[:50]]
\`\`\`

**Affinity Score**:

\`\`\`
Affinity = (interactions with user) / (total interactions)

Example:
- User A viewed 100 posts in last week
- 30 posts were from User B (close friend)
- Affinity(A, B) = 30/100 = 0.3

Interactions: Likes, comments, profile visits, messages
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
┌──────────────┐
│ Mobile App   │
│  (User)      │
└──────┬───────┘
       │ GET /feed
       ▼
┌──────────────┐
│Load Balancer │
└──────┬───────┘
       │
       ▼
┌──────────────┐       ┌──────────────┐
│ Feed Service │◀─────▶│    Redis     │
│ (API Server) │       │ (Feed Cache) │
└──────┬───────┘       └──────────────┘
       │
       ├─────────────────────┬─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Post Service │     │User Service  │     │Ranking Service│
│(Create Post) │     │(Friends,     │     │ (Score Posts)│
└──────┬───────┘     │ Followers)   │     └──────────────┘
       │             └──────────────┘
       ▼
┌──────────────┐     ┌──────────────┐
│ Fanout       │────▶│     Kafka    │
│ Service      │     │(Post Events) │
└──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐
│  PostgreSQL  │     │      S3      │
│(Posts, Users)│     │ (Images,     │
└──────────────┘     │  Videos)     │
                     └──────────────┘
\`\`\`

---

## Step 6: Database Schema

### Users Table

\`\`\`sql
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    created_at TIMESTAMP,
    follower_count INT,
    following_count INT
);
\`\`\`

### Posts Table

\`\`\`sql
CREATE TABLE posts (
    post_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    content TEXT,
    media_urls JSON,  -- ["https://s3.../image1.jpg", ...]
    created_at TIMESTAMP,
    likes_count INT DEFAULT 0,
    comments_count INT DEFAULT 0,
    shares_count INT DEFAULT 0,
    INDEX idx_user_time (user_id, created_at)
);
\`\`\`

### Friendships/Followers Table

\`\`\`sql
CREATE TABLE followers (
    user_id BIGINT,  -- Being followed
    follower_id BIGINT,  -- Following
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, follower_id),
    INDEX idx_follower (follower_id)  -- For "who do I follow?"
);
\`\`\`

### Interactions Table (For Affinity)

\`\`\`sql
CREATE TABLE interactions (
    user_id BIGINT,
    target_user_id BIGINT,
    interaction_type VARCHAR(20),  -- "like", "comment", "view"
    created_at TIMESTAMP,
    INDEX idx_user_time (user_id, created_at)
);
\`\`\`

---

## Step 7: Feed Cache (Redis)

**Structure**:

\`\`\`
Key: feed:{user_id}
Type: List (ordered by timestamp)
Value: [post_id_1, post_id_2, ..., post_id_1000]
TTL: 24 hours

Example:
feed:123 → [98765, 98764, 98763, ..., 97766]
\`\`\`

**Operations**:

\`\`\`
# Add new post to follower's feed
LPUSH feed:123 98765  # Prepend (most recent first)
LTRIM feed:123 0 999  # Keep only 1000 posts

# Retrieve feed
LRANGE feed:123 0 49  # Get top 50 posts

# Remove old posts (cleanup job)
LTRIM feed:123 0 999  # Periodic trimming
\`\`\`

**Memory**:
- 500M active users × 1000 posts × 8 bytes (post ID) = 4 TB
- Redis Cluster: 100 nodes × 40 GB = 4 TB

---

## Step 8: Fanout Service (Message Queue)

**Problem**: User with 10K followers posts → 10K feed writes. Can't do synchronously (blocks post creation).

**Solution**: Asynchronous fanout via message queue.

\`\`\`
1. User creates post → Store in database → Return success immediately
2. Publish event to Kafka: {post_id: 98765, user_id: 123, timestamp: ...}
3. Fanout workers consume events → Query followers → Write to feeds
\`\`\`

**Implementation**:

\`\`\`python
# Post Service
def create_post(user_id, content):
    post_id = db.insert("posts", {user_id: user_id, content: content})
    
    # Publish event (non-blocking)
    kafka.produce("post_created", {
        "post_id": post_id,
        "user_id": user_id,
        "timestamp": now()
    })
    
    return {"post_id": post_id, "status": "published"}

# Fanout Worker (consumes Kafka events)
def fanout_worker():
    for event in kafka.consume("post_created"):
        post_id = event["post_id"]
        user_id = event["user_id"]
        
        # Get followers
        followers = db.query("SELECT follower_id FROM followers WHERE user_id = ?", user_id)
        
        # Batch write to Redis (100 followers at a time)
        for batch in chunks(followers, 100):
            pipeline = redis.pipeline()
            for follower_id in batch:
                pipeline.lpush(f"feed:{follower_id}", post_id)
                pipeline.ltrim(f"feed:{follower_id}", 0, 999)
            pipeline.execute()
\`\`\`

**Benefits**:
- ✅ Post creation fast (< 50ms)
- ✅ Fanout happens asynchronously (seconds)
- ✅ Scalable (add more workers for high load)

---

## Step 9: Real-Time Feed Updates (WebSocket)

**Problem**: User scrolling feed. Friend posts new content. How to show without refresh?

**Solution**: WebSocket connection for push notifications.

\`\`\`python
# Client (JavaScript)
const ws = new WebSocket("wss://feed.example.com");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "new_post") {
        showNotification(\`New post from \${ data.author }\`);
        prependPostToFeed(data.post_id);
    }
};

# Server (Fanout Service)
def notify_followers(post_id, user_id):
    followers = get_online_followers(user_id)  # Only users currently online
    for follower_id in followers:
        websocket_connection = get_connection(follower_id)
        if websocket_connection:
            websocket_connection.send({
                "type": "new_post",
                "post_id": post_id,
                "author": user_id
            })
\`\`\`

**Scale**: 10M concurrent connections → Use WebSocket servers (Pusher, Socket.io, custom)

---

## Step 10: Pagination (Infinite Scroll)

**Challenge**: User scrolls down, load more posts.

\`\`\`
Initial load: Get posts 0-49
Scroll down:  Get posts 50-99
Scroll down:  Get posts 100-149
\`\`\`

**Implementation**:

\`\`\`python
def get_feed_paginated(user_id, offset=0, limit=50):
    # Get from Redis cache
    post_ids = redis.lrange(f"feed:{user_id}", offset, offset + limit - 1)
    
    if not post_ids:
        # Cache miss: Fall back to database (pull model)
        posts = get_feed_from_db(user_id, offset, limit)
    else:
        posts = db.query("SELECT * FROM posts WHERE post_id IN (?)", post_ids)
    
    return posts

# Client request
GET /feed?offset=50&limit=50
\`\`\`

**Cursor-Based Pagination** (Better):

\`\`\`
GET /feed?cursor=2023-01-15T10:30:00&limit=50

# Return posts created_at < cursor (older posts)
SELECT * FROM posts WHERE ... AND created_at < '2023-01-15T10:30:00' ORDER BY created_at DESC LIMIT 50;

# Response includes next cursor
{
  "posts": [...],
  "next_cursor": "2023-01-15T09:15:23"
}
\`\`\`

---

## Step 11: Optimizations

### 1. Pre-Generate Feeds (Idle Time)

\`\`\`
- User hasn't opened app in 6 hours
- Background job: Pre-generate feed, store in Redis
- When user opens app: Instant load (feed ready)
\`\`\`

### 2. Partial Fanout

\`\`\`
- User has 10K followers
- Only fanout to 1K most active followers (opened app in last 24 hours)
- Others get feed via pull model
\`\`\`

### 3. Edge Caching (CDN)

\`\`\`
- Cache popular posts (viral content) at edge
- Reduces latency (serve from nearby CDN node)
\`\`\`

### 4. Batch Ranking

\`\`\`
- Don't rank every post in real-time
- Pre-rank top 100 posts for each user (daily batch job)
- Real-time ranking only for recent posts (last 6 hours)
\`\`\`

---

## Step 12: Handling Celebrities

**Problem**: Celebrity with 100M followers posts → 100M fanout writes (hours to complete).

**Solutions**:

**1. No Fanout** (Pull on Read):
- Celebrity posts stored in their timeline
- Followers query celebrity's timeline when opening feed
- Trade-off: Slower feed generation for celebrity followers

**2. Sampled Fanout**:
- Fanout to 1% of followers (1M users)
- Others get via pull model
- Balances freshness and fanout cost

**3. Separate Celebrity Feed**:
- User's feed = Friends feed + Celebrity feed (merged)
- Celebrity feed updated separately (pull model)

---

## Step 13: Monitoring & Metrics

**Key Metrics**:

1. **Feed Latency**: p50, p95, p99 (target < 500ms)
2. **Fanout Time**: Time to complete fanout for a post
3. **Cache Hit Rate**: % of feeds served from Redis
4. **Post Creation Latency**: Time to publish post
5. **Feed Freshness**: Age of oldest post in feed

**Alerts**:
- Feed latency > 500ms (p95) → Scale feed service
- Fanout backlog > 10 minutes → Add fanout workers
- Cache hit rate < 80% → Increase Redis capacity

---

## Trade-offs

**Push vs Pull**:
- Push: Fast read, slow write (fanout overhead)
- Pull: Slow read (query friends), fast write
- **Choice**: Hybrid (push for normal users, pull for celebrities)

**Chronological vs Ranked**:
- Chronological: Simple, predictable, but low engagement
- Ranked: High engagement, but complex (ML models, real-time scoring)
- **Choice**: Ranked (better user retention)

**Real-Time vs Batch**:
- Real-Time: Fresh feed, high compute cost
- Batch: Stale feed (lag), low cost
- **Choice**: Hybrid (real-time for active users, batch for inactive)

---

## Interview Tips

**Clarify**:
1. Scale: Millions or billions of users?
2. Feed type: Chronological or ranked?
3. Media: Text only or images/videos?
4. Real-time: Must be instant or lag acceptable?

**Emphasize**:
1. **Hybrid Fanout**: Push for normal users, pull for celebrities
2. **Redis Feed Cache**: Pre-computed feeds for fast retrieval
3. **Ranking Algorithm**: Recency + engagement + affinity
4. **Asynchronous Fanout**: Kafka + workers (non-blocking post creation)
5. **Pagination**: Cursor-based for infinite scroll

**Common Mistakes**:
- Pure pull model (too slow at scale)
- Pure push model (celebrity problem)
- No caching (database overload)
- Ignoring ranking (chronological only)
- Synchronous fanout (blocks post creation)

**Follow-up Questions**:
- "How to handle deleted posts? (Remove from all feeds where it appears)"
- "How to prevent spam in feed? (ML classifier, user reports)"
- "How to support multiple feed types? (Following, Discover, Groups)"
- "How to handle edited posts? (Update cache, notify viewers)"

---

## Summary

**Core Architecture**: **Hybrid Push-Pull with Redis Cache**

**Components**:
1. **Feed Service**: Retrieves feeds from cache
2. **Post Service**: Creates posts, publishes events
3. **Fanout Service**: Kafka consumers, write to follower feeds
4. **Ranking Service**: Scores posts (recency, engagement, affinity)
5. **Redis Cluster**: Feed cache (4 TB, 100 nodes)
6. **PostgreSQL**: Posts, users, friendships
7. **S3**: Media storage (images, videos)

**Key Decisions**:
- ✅ Hybrid fanout (push < 10K followers, pull ≥ 10K)
- ✅ Redis feed cache (pre-computed, 1000 posts per user)
- ✅ Asynchronous fanout (Kafka + workers)
- ✅ Ranking by engagement (ML-based scoring)
- ✅ Cursor-based pagination (infinite scroll)
- ✅ WebSocket for real-time updates

**Capacity**:
- 500M DAU
- 100M posts/day
- 58K feed requests/second (120K peak)
- < 500ms feed latency (p95)
- 4 TB Redis cache
- 16 PB/month storage (posts + media)

A production news feed system balances **speed** (< 500ms), **scalability** (billions of users), and **relevance** (ranked posts) to deliver personalized content at massive scale.`,
};

