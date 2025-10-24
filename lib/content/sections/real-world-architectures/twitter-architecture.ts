/**
 * Twitter Architecture Section
 */

export const twitterarchitectureSection = {
  id: 'twitter-architecture',
  title: 'Twitter Architecture',
  content: `Twitter (now X) is a real-time social networking platform where users post short messages (tweets) to their followers. With 330+ million monthly active users posting 500 million tweets per day, Twitter's architecture must handle massive read/write volume, real-time delivery, and global scale. This section explores the technical systems behind Twitter.

## Overview

Twitter's architecture handles unique challenges:
- **500 million tweets** per day
- **330+ million monthly active users**
- **1 billion+ timeline views** per day
- **Real-time delivery**: Tweets appear instantly in followers' timelines
- **Celebrity accounts**: Some users have 100+ million followers
- **Spiky traffic**: Major events cause massive traffic surges

### Key Architectural Challenges

1. **Fanout problem**: When celebrity tweets, write to 100M+ timelines?
2. **Real-time delivery**: Tweets must appear instantly
3. **Scale**: Billions of timeline reads per day
4. **Global distribution**: Users worldwide expect low latency
5. **Hot keys**: Trending topics cause database hotspots

---

## Evolution of Twitter's Architecture

### Phase 1: Monolithic Ruby on Rails (2006-2010)

Early Twitter was a simple Rails monolith with MySQL.

\`\`\`
Browser → Rails App → MySQL Database
\`\`\`

**Problems at Scale**:
- **Fail whale**: Frequent downtime during traffic spikes
- **Slow timelines**: Timeline generation took seconds
- **Database overload**: MySQL couldn't handle write volume

**Famous Incident**: 2008 - Twitter down or slow most of the time.

---

### Phase 2: Service-Oriented Architecture (2010-2013)

Twitter decomposed monolith into services.

**Key Services**:
- **Tweet Service**: Store and serve tweets
- **Timeline Service**: Generate user timelines
- **User Service**: User profiles and relationships
- **Search Service**: Tweet search
- **Notification Service**: Mentions, retweets notifications

**Technology Shift**:
- **Ruby → Scala/Java**: JVM for performance
- **MySQL → Custom Data Stores**: Manhattan (distributed KV store), Cassandra
- **Built Internal Tools**: Finagle (RPC), Gizzard (sharding)

---

### Phase 3: Timeline Fanout Optimization (2013-present)

Twitter optimized timeline generation with hybrid fanout approach.

**Problem**: Generating timeline on-demand (fanout-on-read) was too slow.

**Solution**: Pre-compute timelines (fanout-on-write) with exceptions for celebrities.

---

## Core Components

### 1. Tweet Storage and Serving

Tweets are the core data object: 280 characters + metadata.

**Data Model**:
\`\`\`
Tweet:
- tweet_id (64-bit snowflake ID)
- user_id
- text (280 chars)
- created_at
- mentions ([@user1, @user2])
- hashtags ([#tech, #news])
- media_urls ([image1, video1])
- retweet_count
- like_count
- reply_count
\`\`\`

**Snowflake IDs**:

Twitter uses Snowflake, a distributed ID generation system, for tweet IDs.

**Why not auto-increment?**:
- Auto-increment requires coordination (single point of failure)
- Hard to shard

**Snowflake ID Structure** (64 bits):
\`\`\`
1 bit: unused (always 0)
41 bits: timestamp (milliseconds since epoch)
10 bits: machine ID (datacenter + worker)
12 bits: sequence number (counter per machine per millisecond)

Example: 1234567890123456789
Decode: timestamp=1234567890123, machine=45, sequence=789

Benefits:
- Time-ordered (sortable by time)
- Roughly time-sortable (newer IDs > older IDs)
- Unique across all machines
- No coordination needed
- Can generate 4096 IDs per machine per millisecond
\`\`\`

**Storage**:

**Manhattan** (Twitter's distributed K/V store):
- Built on RocksDB (log-structured merge tree)
- Multi-region replication
- Tunable consistency

\`\`\`
Key: tweet_id
Value: serialized tweet object (Thrift or Protobuf)
\`\`\`

**Caching**:
- **Redis** for hot tweets (trending, popular)
- **Memcached** for user timelines
- Cache hit rate: 95%+

**Tweet Write Flow**:
\`\`\`
Client → API Gateway → Tweet Service
                            ↓
                       Generate tweet_id (Snowflake)
                            ↓
                       Store in Manhattan
                            ↓
                       Publish to Timeline Service (fanout)
                            ↓
                       Index in Search Service
                            ↓
                       Return tweet_id to client
\`\`\`

---

### 2. Timeline Generation (Fanout Architecture)

Timeline is a user's feed of tweets from accounts they follow.

**Challenge**: User follows 1,000 accounts, each tweets 10 times/day = 10,000 potential tweets. How to select and rank?

**Approaches**:

### Fanout-on-Write (Push Model)

When user tweets, push to all followers' timelines immediately.

**Process**:
\`\`\`
User A (100 followers) tweets
    ↓
Timeline Service reads A's followers: [B, C, D, ..., Z]
    ↓
For each follower:
    Push tweet to follower's timeline (Redis list)
    Timeline:B = [tweet_new, tweet_old1, tweet_old2, ...]
\`\`\`

**Pros**:
- **Fast reads**: Timeline pre-computed, just fetch from cache
- **Low latency**: User opens app, timeline loads instantly (<50ms)

**Cons**:
- **Slow writes**: User with 100M followers → 100M writes!
- **Storage**: Each follower stores duplicate timeline
- **Celebrity problem**: Lady Gaga tweets → system overloaded

**Data Structure** (Redis):
\`\`\`
Key: timeline:user:123
Value: [tweet_id_1, tweet_id_2, ..., tweet_id_1000]  (sorted list, newest first)
\`\`\`

---

### Fanout-on-Read (Pull Model)

When user requests timeline, fetch tweets from followed accounts on-demand.

**Process**:
\`\`\`
User B requests timeline
    ↓
Timeline Service reads B's following: [A, C, D, ...]
    ↓
For each followed account:
    Fetch recent tweets (last 24 hours)
    ↓
Merge tweets from all accounts, sort by time
    ↓
Return top 50 tweets
\`\`\`

**Pros**:
- **Fast writes**: No fanout, just store tweet
- **No storage waste**: No duplicate timelines
- **Works for celebrities**: No 100M writes

**Cons**:
- **Slow reads**: Must query 1,000 accounts, merge, sort (takes seconds)
- **High load**: Every timeline request queries many accounts

---

### Hybrid Approach (Twitter's Solution)

Twitter uses a **hybrid** approach to get benefits of both:

**Regular Users (< 1M followers)**: Fanout-on-write
- When they tweet, push to followers' timelines
- Followers get fast reads

**Celebrities (> 1M followers)**: Fanout-on-read
- When they tweet, DON'T push to followers
- When follower requests timeline, fetch celebrity tweets on-demand

**Timeline Read Flow**:
\`\`\`
User B requests timeline:
    ↓
1. Fetch pre-computed timeline from Redis (fanout-on-write tweets)
    timeline:user:B → [tweet1, tweet2, ..., tweet50]
    ↓
2. Check if B follows any celebrities (stored in B's profile)
    following_celebrities: [CelebrityA, CelebrityB]
    ↓
3. Fetch recent tweets from these celebrities (fanout-on-read)
    CelebrityA's recent tweets: [tweetA1, tweetA2]
    CelebrityB's recent tweets: [tweetB1, tweetB2]
    ↓
4. Merge all tweets, sort by time
    [tweetA1, tweet1, tweetB1, tweet2, ...]
    ↓
5. Return top 50
\`\`\`

**Benefits**:
- **Fast reads for most users**: Pre-computed timeline (95% of users)
- **Handles celebrities**: No 100M writes
- **Scalable**: Can adjust threshold (e.g., 100K followers)

**Trade-Offs**:
- **Slight latency for celebrity followers**: Must fetch on-demand (adds 20-50ms)
- **Complexity**: Two code paths to maintain

---

### 3. Social Graph (Relationships)

Twitter's social graph stores following relationships: user A follows user B.

**Graph Structure**:
\`\`\`
User A follows: [B, C, D, ...]  (outgoing edges)
User B followed by: [A, E, F, ...]  (incoming edges)
\`\`\`

**Storage**: **FlockDB** (Twitter's graph database)

**FlockDB Design**:
- Custom-built for Twitter's needs
- Stores directed edges (A → B)
- Optimized for:
  - Fast writes (follow/unfollow)
  - Fast reads (get followers, get following)
  - Sharded by user_id

**Data Model**:
\`\`\`
Table: follows
Shard Key: follower_id
Columns: followee_id, timestamp

Table: followers (reverse index)
Shard Key: followee_id
Columns: follower_id, timestamp
\`\`\`

**Queries**:
- "Who does user A follow?" → Query follows table, shard by A
- "Who follows user B?" → Query followers table, shard by B
- "Does A follow B?" → Query follows table (A → B edge)

**Challenges**:

**1. Followers Count**:
- Problem: User with 100M followers → Count is expensive
- Solution: Maintain counter (updated asynchronously)

**2. Mutual Followers**:
- Problem: Show "You and 5 others follow @user"
- Solution: Set intersection (A's following ∩ B's followers)
- Optimization: Sample (not exact count, just "5+")

**3. Follow Recommendations**:
- Problem: Suggest accounts to follow
- Solution: Machine learning (based on your follows, interests, activity)

---

### 4. Search

Twitter Search allows users to find tweets by keyword, hashtag, or user.

**Search Index**: **Earlybird** (Twitter's real-time search engine)

**Built on Lucene**:
- Inverted index (keyword → tweet IDs)
- Distributed across shards
- Optimized for real-time updates

**Index Structure**:
\`\`\`
Keyword: "machine learning"
Tweets: [tweet1, tweet5, tweet10, ...]

Hashtag: #tech
Tweets: [tweet3, tweet7, tweet15, ...]
\`\`\`

**Indexing Flow**:
\`\`\`
User tweets "Excited about machine learning!"
    ↓
Tweet Service → Search Service
    ↓
Tokenize: ["excited", "machine", "learning"]
    ↓
Index each term: "excited" → tweet123, "machine" → tweet123, "learning" → tweet123
    ↓
Tweet searchable within 5 seconds
\`\`\`

**Search Query**:
\`\`\`
User searches "machine learning"
    ↓
Search Service queries index
    ↓
Retrieve tweet IDs matching both "machine" AND "learning"
    ↓
Rank by relevance (engagement, recency, user's social graph)
    ↓
Return top 20 tweets
\`\`\`

**Ranking Factors**:
- **Recency**: Recent tweets ranked higher
- **Engagement**: Likes, retweets boost rank
- **Social graph**: Tweets from followed accounts ranked higher
- **Verified accounts**: Verified users ranked higher

**Challenges**:
- **Real-time**: Index tweets within seconds
- **Scale**: 500M tweets/day to index
- **Trending topics**: Hot keywords cause query spikes

**Optimization**:
- **Sharding**: Distribute index across 100s of shards (shard by time)
- **Caching**: Cache popular search queries
- **Read replicas**: Multiple replicas for read scalability

---

### 5. Trending Topics and Hashtags

Trending topics show popular hashtags and keywords in real-time.

**How it Works**:

**1. Stream Processing**:
- All tweets streamed to analytics pipeline (Apache Storm / Apache Heron)
- Extract hashtags and keywords
- Count occurrences in sliding time window (e.g., last 15 minutes)

**2. Trend Detection**:
- Compare current count to historical baseline
- If count spikes (e.g., 10x normal), it's trending
- Example: "#WorldCup" normally 1K tweets/15min, now 50K → Trending!

**3. Ranking**:
- Rank trends by:
  - Volume (tweet count)
  - Velocity (rate of increase)
  - Freshness (how recent)

**4. Personalization**:
- Trends customized per user based on:
  - Location (US trends vs Japan trends)
  - Interests (tech, sports, politics)
  - Following (people you follow tweeting about it)

**Data Pipeline**:
\`\`\`
Tweets → Kafka → Stream Processor (Storm/Heron)
                       ↓
                  Count hashtags per 15-min window
                       ↓
                  Detect spikes (compare to baseline)
                       ↓
                  Store in Redis (trending:global, trending:location:US, ...)
                       ↓
                  API serves trends to clients
\`\`\`

**Challenges**:
- **Spam**: Bots spamming hashtags to manipulate trends
- **Solution**: ML models detect spam, filter out bot tweets
- **Regional trends**: Different countries have different trends
- **Solution**: Shard by location (geo-tagging tweets)

---

### 6. Notifications

Twitter notifies users about mentions, replies, likes, retweets, follows.

**Notification Types**:
- **@mentions**: Someone mentioned you in a tweet
- **Replies**: Someone replied to your tweet
- **Likes**: Someone liked your tweet
- **Retweets**: Someone retweeted your tweet
- **Follows**: Someone followed you
- **Direct messages**: Someone sent you a DM

**Architecture**:

**Event-Driven**:
\`\`\`
User A likes User B's tweet
    ↓
Like Service publishes event: "user_A_liked_tweet_B"
    ↓
Notification Service consumes event
    ↓
Check User B's notification preferences (push, email, none)
    ↓
If push enabled:
    Send push notification to User B's device (via APNs / FCM)
    ↓
Store notification in database (for in-app notification tab)
\`\`\`

**Notification Delivery**:

**1. Push Notifications**:
- Mobile: Apple Push Notification Service (APNs), Firebase Cloud Messaging (FCM)
- Web: Web Push API

**2. In-App Notifications**:
- Stored in database (Manhattan)
- Displayed in notification tab when user opens app

**Data Model**:
\`\`\`
Table: notifications
Partition Key: user_id
Clustering Key: timestamp DESC
Columns: type, actor_id, tweet_id, read_status

Query: "Get unread notifications for user X" → Single partition, filter by read_status=false
\`\`\`

**Notification Preferences**:
- Users can mute notifications from specific accounts
- Disable certain notification types
- Quiet hours (no notifications 10 PM - 8 AM)

**Challenges**:
- **Volume**: Popular tweet gets 100K likes → 100K notification events
- **Deduplication**: User likes, then unlikes, then likes again → Send 1 notification, not 3
- **Aggregation**: "5 people liked your tweet" instead of 5 separate notifications

---

### 7. Direct Messages

Twitter DMs is a real-time messaging system.

**Requirements**:
- **Low latency**: Messages delivered instantly
- **Persistent storage**: Message history
- **Read receipts**: Seen indicators
- **Media**: Photos, videos, GIFs

**Architecture**:

**WebSocket Connection**:
- Persistent connection for real-time message delivery
- User opens Twitter → Establishes WebSocket → Receives messages instantly

**Message Storage** (Manhattan):
\`\`\`
Table: messages
Partition Key: conversation_id
Clustering Key: timestamp DESC
Columns: sender_id, receiver_id, text, media_url, message_id
\`\`\`

**Message Flow**:
\`\`\`
User A sends message to User B
    ↓
Client → WebSocket → DM Service
    ↓
Store message in Manhattan (conversation_id)
    ↓
Look up User B's connected WebSocket server (via Redis)
    ↓
Forward message to User B's WebSocket
    ↓
User B receives message instantly
    ↓
If User B offline:
    Store as unread, send push notification
\`\`\`

**Read Receipts**:
- When User B reads message, send "message_read" event
- Update message status in database
- Notify User A via WebSocket

**Challenges**:
- **Spam**: Unsolicited DMs
- **Solution**: Filters (only allow DMs from followed accounts)
- **Message requests**: Separate inbox for non-followed accounts

---

## Technology Stack

### Custom-Built Systems

**1. Manhattan (Distributed Key-Value Store)**:
- Twitter's primary datastore
- Built on RocksDB (LSM tree)
- Multi-region replication
- Strong consistency within region, eventual consistency across regions

**2. Finagle (RPC Framework)**:
- Scala-based RPC framework
- Built on Netty (non-blocking I/O)
- Features: Load balancing, circuit breakers, retries, timeouts
- Used for service-to-service communication

**3. Gizzard (Sharding Framework)**:
- Application-level sharding
- Shard management (add, remove, rebalance)
- Replication (master-slave)

**4. FlockDB (Graph Database)**:
- Custom graph database for social graph
- Optimized for Twitter's access patterns
- Sharded by user_id

**5. Snowflake (ID Generation)**:
- Distributed unique ID generator
- Time-ordered IDs
- No coordination required

**6. Earlybird (Real-Time Search)**:
- Built on Apache Lucene
- Real-time indexing (tweets searchable within seconds)
- Distributed across shards

---

### Open-Source Technologies

- **Apache Kafka**: Event streaming
- **Apache Hadoop**: Batch processing, analytics
- **Apache Storm / Heron**: Stream processing
- **Redis**: Caching, timelines
- **Memcached**: Caching
- **MySQL**: Some legacy data
- **Cassandra**: Time-series data, analytics

---

## Scaling Challenges and Solutions

### Challenge 1: Fail Whale (Downtime)

**Problem**: Early Twitter (2008-2010) frequently crashed during traffic spikes.

**Root Cause**: Monolithic Rails app, single MySQL instance.

**Solutions**:
1. **Decompose to microservices**: Independent scaling
2. **Sharding**: Distribute data across multiple databases
3. **Caching**: Redis/Memcached reduce database load
4. **Rate limiting**: Prevent abuse, protect infrastructure

**Result**: Twitter now highly available (99.9%+ uptime).

---

### Challenge 2: Celebrity Tweets (Fanout Problem)

**Problem**: Lady Gaga tweets → 100M followers → 100M timeline writes → System overwhelmed.

**Solution**: Hybrid fanout (explained earlier).

**Result**: Writes complete in <100ms even for celebrities.

---

### Challenge 3: World Cup / Events (Traffic Spike)

**Problem**: Major events cause 10x traffic spikes.

**Solutions**:
1. **Auto-scaling**: Automatically add servers during spikes
2. **Caching**: Heavily cache popular content
3. **Rate limiting**: Prioritize read over write during peak
4. **Load shedding**: Drop non-critical requests (e.g., analytics)

**Result**: Twitter handles Super Bowl, World Cup without downtime.

---

### Challenge 4: Search Latency

**Problem**: Searching 500M tweets/day is slow.

**Solutions**:
1. **Sharding**: Distribute index across 100s of shards (shard by time)
2. **Caching**: Cache popular search queries
3. **Read replicas**: Multiple replicas for read scaling
4. **Ranking optimization**: Pre-compute ranking signals

**Result**: P99 search latency <200ms.

---

## Key Lessons

### 1. Pre-Compute When Possible

Timeline fanout-on-write pre-computes timelines for fast reads. Trade write complexity for read speed.

### 2. Hybrid Approaches for Edge Cases

Celebrity users require different handling. Don't treat all users the same.

### 3. Custom-Built Systems for Core Needs

Twitter built Manhattan, Finagle, Gizzard, FlockDB, Snowflake for specific needs. Off-the-shelf solutions didn't meet requirements.

### 4. Real-Time Requires Streaming

Twitter uses Storm/Heron for real-time analytics (trending topics, spam detection). Batch processing too slow.

### 5. Observability is Critical

With 100s of microservices, distributed tracing (Zipkin) and metrics (Prometheus) are essential.

---

## Interview Tips

**Q: How would you design Twitter's timeline?**

A: Use hybrid fanout approach. For users with <1M followers, use fanout-on-write: when user tweets, write to all followers' timelines (Redis list) asynchronously. For users with >1M followers, use fanout-on-read: don't write to followers, fetch on-demand when followers request timeline. When user requests timeline: (1) Fetch pre-computed timeline from Redis (fanout-on-write tweets). (2) Check if user follows any celebrities, fetch their recent tweets. (3) Merge both sources, sort by time. (4) Return top 50. Cache result for 1-2 minutes. Benefits: Fast reads (pre-computed for most users), handles celebrities (no 100M writes), scalable.

**Q: How does Twitter generate unique IDs at scale?**

A: Twitter uses Snowflake, a distributed ID generator. Each ID is 64 bits: 1 bit unused, 41 bits timestamp (milliseconds since epoch), 10 bits machine ID (datacenter + worker), 12 bits sequence number (counter per machine per millisecond). Benefits: (1) Time-ordered (sortable by creation time). (2) Unique across all machines (machine ID ensures uniqueness). (3) No coordination (each machine generates independently). (4) High throughput (4096 IDs per machine per millisecond). (5) Decode timestamp from ID (useful for analytics). Each datacenter has 32 machines, each machine generates 4096 IDs/ms = 128K IDs/ms per datacenter.

**Q: How does Twitter handle trending topics?**

A: Stream all tweets to analytics pipeline (Apache Storm/Heron). Extract hashtags and keywords. Count occurrences in sliding 15-minute window using approximate counting (HyperLogLog). Compare current count to historical baseline (last 24 hours average). If count spikes 10x, mark as trending. Rank trends by: (1) Volume (tweet count). (2) Velocity (rate of increase). (3) Freshness (how recent). Personalize by location (geo-tagged tweets) and user interests. Filter spam (ML models detect bot activity). Store trending topics in Redis (trending:global, trending:location:US). API serves to clients, refreshed every 30 seconds.

---

## Summary

Twitter's architecture demonstrates building a real-time social platform at massive scale:

**Key Takeaways**:

1. **Hybrid fanout**: Fanout-on-write for regular users, fanout-on-read for celebrities
2. **Custom systems**: Built Manhattan, Finagle, FlockDB, Snowflake for specific needs
3. **Real-time indexing**: Tweets searchable within seconds (Earlybird)
4. **Stream processing**: Storm/Heron for trending topics, spam detection
5. **Graph database**: FlockDB optimized for social graph queries
6. **Snowflake IDs**: Distributed ID generation, time-ordered, no coordination
7. **Microservices**: Decomposed monolith for independent scaling
8. **Caching everywhere**: Redis/Memcached at multiple layers

Twitter's evolution from monolith to microservices showcases scaling challenges and solutions for real-time social platforms.
`,
};
