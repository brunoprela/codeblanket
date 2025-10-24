/**
 * Design Twitter Section
 */

export const twitterSection = {
  id: 'twitter',
  title: 'Design Twitter',
  content: `Twitter is a social network where users post short messages (tweets), follow other users, and view personalized timelines. Designing Twitter involves handling massive scale, real-time updates, and complex feed generation algorithms.

## Problem Statement

Design a simplified Twitter that supports:
- **Post tweets** (280 characters max)
- **Follow/unfollow** users
- **View timeline** (tweets from followed users)
- **View user profile** (user's tweets)
- **Search tweets** (hashtags, keywords)
- **Retweet and like** (optional)
- **Trending topics** (optional)

**Scale**: 300 million users, 500 million tweets/day, 10 billion timeline views/day

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Post Tweet**: Users can post 280-character text + optional media
2. **Follow System**: Users can follow/unfollow others
3. **Home Timeline**: View tweets from followed users (chronological)
4. **User Timeline**: View user's own tweets
5. **Retweet**: Share others' tweets
6. **Like**: Like tweets
7. **Search**: Search tweets by keywords, hashtags
8. **Trending**: Display trending hashtags/topics

### Non-Functional Requirements

1. **High Availability**: 99.99% uptime
2. **Low Latency**: Timeline loads in < 200ms
3. **Scalable**: Handle 300M users, 500M tweets/day
4. **Eventual Consistency**: Timeline can be slightly stale
5. **Real-time**: Tweets appear within seconds of posting

### Out of Scope

- Direct messages
- Notifications
- Video uploads
- Polls, Spaces, etc.

---

## Step 2: Capacity Estimation

### Traffic

**Write load (Tweets)**:
- 500M tweets/day = ~5,800 tweets/sec
- Peak: ~10,000 tweets/sec

**Read load (Timeline views)**:
- 10B timeline views/day = ~115,000 reads/sec
- Peak: ~200,000 reads/sec
- **Read:Write ratio = 20:1**

### Storage

**Tweets**:
- 500M tweets/day
- Average tweet: 300 bytes (text + IDs + timestamps)
- Media stored separately (S3)
- 500M × 300 bytes = 150 GB/day
- 5 years: 150 GB × 365 × 5 = 274 TB

**Users**:
- 300M users
- Per user: 1 KB (name, bio, profile_url, etc.)
- Total: 300 GB

**Follow relationships**:
- Average 200 follows per user
- 300M users × 200 = 60 billion relationships
- Each relationship: 16 bytes (follower_id, followee_id)
- 60B × 16 bytes = 960 GB ≈ 1 TB

### Bandwidth

**Write bandwidth**:
- 5,800 tweets/sec × 300 bytes = 1.7 MB/sec (~14 Mbps)

**Read bandwidth**:
- 115,000 reads/sec × 10 tweets × 300 bytes = 345 MB/sec (~2.7 Gbps)

### Cache

Cache hot users' timelines:
- 10B timeline requests/day
- Assume 20M active users
- Cache 5M hottest users' timelines (top 25%)
- Each timeline: 500 tweets × 300 bytes = 150 KB
- 5M × 150 KB = 750 GB cache

---

## Step 3: System API Design

### POST /api/v1/tweets

\`\`\`json
Request:
{
  "text": "Hello World!",
  "media_urls": ["https://cdn.twitter.com/img1.jpg"],
  "user_id": 123
}

Response (201):
{
  "tweet_id": "17389234823",
  "created_at": "2024-10-24T10:00:00Z",
  "user_id": 123
}
\`\`\`

### GET /api/v1/timeline/home

\`\`\`json
Request: GET /api/v1/timeline/home?user_id=123&page=1&limit=20

Response (200):
{
  "tweets": [
    {
      "tweet_id": "17389234823",
      "user_id": 456,
      "username": "alice",
      "text": "Hello World!",
      "created_at": "2024-10-24T10:00:00Z",
      "likes": 42,
      "retweets": 5
    }
  ],
  "next_cursor": "abc123"
}
\`\`\`

### POST /api/v1/follow

\`\`\`json
Request:
{
  "follower_id": 123,
  "followee_id": 456
}

Response (201):
{
  "status": "followed"
}
\`\`\`

---

## Step 4: Database Schema

### Users Table

\`\`\`sql
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    bio TEXT,
    profile_image_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    follower_count INT DEFAULT 0,
    following_count INT DEFAULT 0,
    INDEX idx_username (username)
);
\`\`\`

### Tweets Table

\`\`\`sql
CREATE TABLE tweets (
    tweet_id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    text VARCHAR(280) NOT NULL,
    media_urls JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    like_count INT DEFAULT 0,
    retweet_count INT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created_at (created_at)
);
\`\`\`

### Follows Table

\`\`\`sql
CREATE TABLE follows (
    follower_id BIGINT NOT NULL,
    followee_id BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_follower (follower_id),
    INDEX idx_followee (followee_id)
);
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
┌─────────────────┐         ┌─────────────────┐
│   Load Balancer │         │   CDN (Media)   │
└────────┬────────┘         └─────────────────┘
         │                           ▲
         │                           │ (Images/Videos)
         ▼                           │
┌─────────────────┐         ┌─────────────────┐
│  API Gateway    │────────▶│   S3 Storage    │
└────────┬────────┘         └─────────────────┘
         │
    ┌────┴────┬───────────────┬─────────────┐
    │         │               │             │
    ▼         ▼               ▼             ▼
┌────────┐ ┌────────┐  ┌──────────┐  ┌──────────┐
│ Tweet  │ │Timeline│  │ Follow   │  │ Search   │
│Service │ │Service │  │ Service  │  │ Service  │
└───┬────┘ └───┬────┘  └────┬─────┘  └────┬─────┘
    │          │             │             │
    │          ▼             │             ▼
    │    ┌──────────┐        │      ┌─────────────┐
    │    │  Redis   │        │      │Elasticsearch│
    │    │  Cache   │        │      └─────────────┘
    │    └──────────┘        │
    │                        │
    ▼                        ▼
┌────────────────────────────────────┐
│    Primary Database (MySQL)         │
│  (Users, Tweets, Follows)           │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Read Replicas (10+)                │
└────────────────────────────────────┘

         ┌──────────────┐
         │ Message Queue│
         │   (Kafka)    │
         └──────┬───────┘
                │
         ┌──────▼────────┐
         │ Fanout Workers│
         └───────────────┘
\`\`\`

---

## Step 6: Core Challenge - Feed Generation

**The Problem**: User Alice follows 500 people. How do we generate her home timeline?

### Approach 1: Fanout-on-Read (Pull Model)

**Algorithm**:
1. User requests timeline
2. Query: Get Alice's followings (500 IDs)
3. Query: Get recent tweets from each followee
4. Merge and sort tweets by timestamp
5. Return top 20 tweets

**SQL Query**:
\`\`\`sql
SELECT t.* FROM tweets t
JOIN follows f ON t.user_id = f.followee_id
WHERE f.follower_id = {alice_id}
AND t.created_at > NOW() - INTERVAL 3 DAYS
ORDER BY t.created_at DESC
LIMIT 20;
\`\`\`

**Pros**:
- ✅ No fanout work on tweet creation (fast writes)
- ✅ Minimal storage (no pre-computed timelines)
- ✅ Always fresh (real-time data)

**Cons**:
- ❌ Slow reads (must query + merge 500 users' tweets)
- ❌ Database load (heavy joins)
- ❌ Cannot scale to millions of timeline requests

### Approach 2: Fanout-on-Write (Push Model)

**Algorithm**:
1. User Bob posts tweet
2. Get Bob's followers list (say, 10,000 followers)
3. Insert tweet into each follower's pre-computed timeline
4. User Alice requests timeline → Read from pre-computed cache/table

**Data Structure**:
\`\`\`sql
CREATE TABLE timelines (
    user_id BIGINT,
    tweet_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, created_at, tweet_id),
    INDEX idx_user_created (user_id, created_at)
);
\`\`\`

**Pros**:
- ✅ Fast reads (pre-computed, indexed by user_id)
- ✅ Scales to millions of reads/sec
- ✅ Low database load during reads

**Cons**:
- ❌ Slow writes (must fanout to 10K followers)
- ❌ Celebrity problem: Posting a tweet fans out to 100M followers (infeasible)
- ❌ Storage explosion: Duplicate tweet_id stored billions of times

### Approach 3: Hybrid (Twitter's Production Approach)

**Strategy**:
- **Regular users** (< 10K followers): Fanout-on-write
- **Celebrities** (> 10K followers): Fanout-on-read
- **Active users**: Keep timeline in Redis cache
- **Inactive users**: Compute on-demand

**Algorithm**:
\`\`\`
1. User requests timeline
2. Check Redis cache:
   - Cache HIT → Return cached timeline
   - Cache MISS → Compute timeline

3. Compute timeline:
   a. Read pre-computed timeline (fanout-on-write from regular users)
   b. Query tweets from celebrities user follows (fanout-on-read)
   c. Merge both lists, sort by timestamp
   d. Cache result in Redis (TTL: 5 minutes)
4. Return timeline
\`\`\`

**Thresholds**:
- Followers < 10K → Fanout-on-write
- Followers >= 10K → No fanout (read at timeline gen time)

**Benefits**:
- ✅ Fast writes for regular users
- ✅ Fast reads for all users (cached)
- ✅ Handles celebrity problem
- ✅ Scalable and practical

---

## Step 7: Tweet Creation Flow

**Flow**:
\`\`\`
1. User posts tweet via POST /api/v1/tweets
2. Tweet Service:
   a. Validate text (<= 280 characters)
   b. If media → Upload to S3 via Media Service
   c. Insert tweet into Tweets table → Get tweet_id
3. Publish tweet_id to Kafka topic "new_tweets"
4. Return tweet_id to user (async fanout happens in background)

--- Background Fanout Worker ---

5. Fanout Worker consumes "new_tweets" from Kafka
6. Query: Get user's follower count
7. If follower_count < 10K:
   a. Query: Get all followers (follower_ids)
   b. For each follower:
      - Insert into timelines table: (follower_id, tweet_id, created_at)
      - Or add to Redis list: LPUSH timeline:{follower_id} {tweet_id}
8. If follower_count >= 10K:
   - Skip fanout (celebrity, will be fetched on-demand)
\`\`\`

**Why Kafka?**
- Decouples tweet creation from fanout
- Fanout can lag (eventual consistency OK)
- Retries if fanout worker fails

---

## Step 8: Timeline Generation Flow

**Flow**:
\`\`\`
1. User requests GET /api/v1/timeline/home?user_id=alice
2. Check Redis cache: GET timeline:alice
   → If cache HIT: Return cached timeline (< 5 minutes old)
3. If cache MISS:
   a. Query timelines table for pre-computed tweets (fanout-on-write)
   b. Query follows table for celebrities user follows
   c. Query tweets table for recent tweets from celebrities
   d. Merge both lists, sort by created_at DESC
   e. Limit to top 100 tweets
   f. Store in Redis: SETEX timeline:alice 300 {tweets}
4. Hydrate tweet details:
   - Batch query: Get user info, media URLs, like counts
5. Return timeline to user
\`\`\`

**Redis Cache Structure**:
\`\`\`
Key: timeline:{user_id}
Value: [tweet_id1, tweet_id2, ..., tweet_id100]
TTL: 300 seconds (5 minutes)
\`\`\`

---

## Step 9: Database Sharding

### Sharding Strategy

**Tweets Table**: Shard by \`user_id\`
- Each user's tweets go to same shard
- User timeline query hits single shard
- Formula: \`shard = user_id % NUM_SHARDS\`

**Follows Table**: Shard by \`follower_id\`
- Query "Who does Alice follow?" hits single shard
- Query "Who follows Bob?" requires scatter-gather (rare operation)

**Users Table**: Shard by \`user_id\`

### Example (4 shards):

\`\`\`
User 123 → Shard 3 (123 % 4)
User 456 → Shard 0 (456 % 4)
\`\`\`

---

## Step 10: Additional Features

### Likes

**Table**:
\`\`\`sql
CREATE TABLE likes (
    user_id BIGINT,
    tweet_id BIGINT,
    created_at TIMESTAMP,
    PRIMARY KEY (user_id, tweet_id)
);
\`\`\`

**Flow**:
1. User likes tweet → Insert into likes table
2. Increment like_count asynchronously (Redis counter + batch sync)

### Retweets

**Strategy**: Store as new tweet with \`original_tweet_id\` field
\`\`\`sql
ALTER TABLE tweets ADD COLUMN original_tweet_id BIGINT NULL;
\`\`\`

### Search

**Use Elasticsearch**:
- Index tweets in Elasticsearch
- Support full-text search, hashtag search
- Update index via Kafka consumer (eventual consistency)

**Query**:
\`\`\`
POST /tweets/_search
{
  "query": {
    "match": {
      "text": "machine learning"
    }
  },
  "sort": [{"created_at": "desc"}]
}
\`\`\`

### Trending Topics

**Algorithm**:
1. Track hashtag mentions in sliding window (last 1 hour)
2. Use Redis sorted set: \`ZINCRBY trending:hourly #{hashtag} 1\`
3. Query top 10: \`ZREVRANGE trending:hourly 0 9 WITHSCORES\`
4. Background job resets hourly

---

## Step 11: Optimizations

### 1. Timeline Caching

Cache in Redis with short TTL (5 mins):
- Reduces database load by 95%
- Acceptable staleness for social media

### 2. Tweet Denormalization

Store user info with tweet for fast rendering:
\`\`\`json
{
  "tweet_id": "123",
  "text": "Hello",
  "user": {
    "user_id": "456",
    "username": "alice",
    "profile_image": "url"
  }
}
\`\`\`

Avoids extra user lookup per tweet.

### 3. CDN for Media

Store images/videos in S3, serve via CloudFront CDN.

### 4. Rate Limiting

Prevent spam:
- Max 100 tweets/hour per user
- Max 100 follows/hour per user

---

## Trade-offs

### Consistency vs Availability

**Choice**: **Availability** (AP system)
- Timeline can be slightly stale
- Follow/unfollow eventually consistent
- Acceptable for social media

### Fanout Trade-offs

| Approach | Write Speed | Read Speed | Storage | Scalability |
|----------|-------------|------------|---------|-------------|
| Fanout-on-read | Fast | Slow | Low | Poor |
| Fanout-on-write | Slow | Fast | High | Good |
| Hybrid | Medium | Fast | Medium | Excellent |

**Production choice**: Hybrid

---

## Interview Tips

### What to Clarify

1. **Scale**: How many users? Tweets per day?
2. **Features**: Retweets? Likes? DMs?
3. **Latency**: What's acceptable timeline load time?
4. **Consistency**: Is eventual consistency OK?

### What to Emphasize

1. **Feed generation**: Explain fanout-on-write vs fanout-on-read
2. **Celebrity problem**: Hybrid approach
3. **Caching**: Redis for timelines
4. **Sharding**: By user_id

### Common Mistakes

1. ❌ Using fanout-on-write for all users (celebrity problem)
2. ❌ Real-time consistency requirements (unrealistic at scale)
3. ❌ Not caching timelines (database overload)
4. ❌ Ignoring write amplification in fanout-on-write

### Follow-up Questions

- "How do you handle trending topics?"
- "How would you implement notifications?"
- "What if Elon Musk with 100M followers posts a tweet?"
- "How do you prevent spam and bots?"

---

## Summary

**Core Components**:
1. **Tweet Service**: Handle tweet creation, fanout worker
2. **Timeline Service**: Generate personalized timelines
3. **Follow Service**: Manage follow relationships
4. **Kafka**: Async fanout pipeline
5. **Redis**: Timeline cache (750 GB)
6. **MySQL (Sharded)**: Users, Tweets, Follows tables
7. **Elasticsearch**: Tweet search

**Key Decisions**:
- ✅ Hybrid fanout: Write for normal users, read for celebrities
- ✅ Redis caching with 5-minute TTL
- ✅ Eventual consistency acceptable
- ✅ Shard by user_id
- ✅ Kafka for async fanout

**Capacity**:
- 500M tweets/day (5.8K tweets/sec)
- 10B timeline views/day (115K reads/sec)
- 274 TB storage over 5 years
- 750 GB Redis cache

This design handles **Twitter-scale traffic** with **sub-200ms timeline loads** using battle-tested patterns from real-world social networks.`,
};
