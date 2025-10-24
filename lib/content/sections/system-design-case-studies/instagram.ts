/**
 * Design Instagram Section
 */

export const instagramSection = {
  id: 'instagram',
  title: 'Design Instagram',
  content: `Instagram is a photo and video sharing social network where users post media, follow others, and interact through likes and comments. The key challenge is handling massive media storage and efficient feed generation at scale.

## Problem Statement

Design Instagram with the following features:
- **Upload photos/videos** with captions and hashtags
- **Follow/unfollow** users
- **News feed** (photos from followed users)
- **User profile** (user's photos grid)
- **Like and comment** on posts
- **Search** by username, hashtags
- **Stories** (optional)

**Scale**: 1 billion users, 100 million photos/day, 1 billion feed views/day

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Upload Media**: Users can upload photos (max 10 MB) and videos (max 100 MB)
2. **Follow System**: Users can follow/unfollow others
3. **News Feed**: View chronological photos from followed users
4. **User Profile**: Grid view of user's photos
5. **Likes & Comments**: Interact with posts
6. **Search**: Search users, hashtags
7. **Direct Messaging**: Basic chat (optional)

### Non-Functional Requirements

1. **High Availability**: 99.9% uptime
2. **Low Latency**: Feed loads in < 300ms, images load in < 500ms
3. **Scalable**: Handle billions of photos, millions of concurrent users
4. **Durable**: Photos must never be lost
5. **Consistency**: Eventual consistency acceptable for feeds
6. **Global**: Low latency worldwide (CDN)

### Out of Scope

- Video editing
- Filters (assume pre-processed)
- Stories, Reels, IGTV
- Shopping

---

## Step 2: Capacity Estimation

### Traffic

**Writes (Photo uploads)**:
- 100M photos/day = ~1,160 uploads/sec
- Peak: ~2,500 uploads/sec

**Reads (Feed views)**:
- 1B feed views/day = ~11,500 views/sec
- Peak: ~25,000 views/sec
- **Read:Write ratio = 10:1**

### Storage

**Photos**:
- 100M photos/day
- Average photo size: 2 MB (after compression)
- Multiple sizes: Thumbnail (50 KB), Medium (500 KB), Original (2 MB)
- Total per photo: 2.5 MB
- 100M × 2.5 MB = 250 TB/day
- 5 years: 250 TB × 365 × 5 = 456 PB (petabytes!)

**Metadata**:
- 100M photos/day × 1 KB (caption, tags, user_id) = 100 GB/day
- 5 years: 100 GB × 365 × 5 = 182 TB

**Users**:
- 1B users × 1 KB = 1 TB

### Bandwidth

**Upload bandwidth**:
- 1,160 uploads/sec × 2 MB = 2.3 GB/sec (~18 Gbps)

**Download bandwidth** (feed views):
- 11,500 views/sec × 10 photos × 500 KB = 57.5 GB/sec (~460 Gbps)
- **CDN is critical**

### Cache

Cache hot photos metadata and thumbnails:
- 1B feed views/day, assume 20% unique photos
- 200M photos × 50 KB (thumbnail) = 10 TB cache

---

## Step 3: System API Design

### POST /api/v1/photos

\`\`\`json
Request:
{
  "user_id": 123,
  "image": "<binary>",
  "caption": "Beautiful sunset!",
  "hashtags": ["#sunset", "#nature"],
  "location": "San Francisco, CA"
}

Response (201):
{
  "photo_id": "a3b4c5d6",
  "url": "https://cdn.instagram.com/photos/a3b4c5d6.jpg",
  "thumbnail_url": "https://cdn.instagram.com/photos/a3b4c5d6_thumb.jpg",
  "created_at": "2024-10-24T10:00:00Z"
}
\`\`\`

### GET /api/v1/feed

\`\`\`json
Request: GET /api/v1/feed?user_id=123&limit=20&cursor=xyz

Response (200):
{
  "photos": [
    {
      "photo_id": "a3b4c5d6",
      "user_id": 456,
      "username": "alice",
      "profile_image": "url",
      "image_url": "url",
      "caption": "Beautiful sunset!",
      "likes_count": 142,
      "created_at": "2024-10-24T10:00:00Z"
    }
  ],
  "next_cursor": "abc123"
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
    post_count INT DEFAULT 0,
    INDEX idx_username (username)
);
\`\`\`

### Photos Table

\`\`\`sql
CREATE TABLE photos (
    photo_id VARCHAR(20) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    caption TEXT,
    image_urls JSON,  -- {thumbnail, medium, original}
    hashtags JSON,
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    like_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_created_at (created_at)
);
\`\`\`

**image_urls JSON Example**:
\`\`\`json
{
  "thumbnail": "https://cdn.ig.com/thumb_123.jpg",
  "medium": "https://cdn.ig.com/med_123.jpg",
  "original": "https://cdn.ig.com/orig_123.jpg"
}
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

### Likes Table

\`\`\`sql
CREATE TABLE likes (
    user_id BIGINT NOT NULL,
    photo_id VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, photo_id),
    INDEX idx_photo (photo_id)
);
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
                              ┌───────────────┐
                              │ Load Balancer │
                              └───────┬───────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│Upload Service│            │ Feed Service │            │ API Gateway  │
└──────┬───────┘            └──────┬───────┘            └──────┬───────┘
       │                           │                           │
       │                           │                           │
       ▼                           ▼                           ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ Image Proc   │            │ Redis Cache  │            │   Database   │
│  Service     │            │  (10 TB)     │            │  (Sharded)   │
└──────┬───────┘            └──────────────┘            └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│       Object Storage (S3)             │
│       Photos: 456 PB over 5 years     │
└───────────┬──────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│       CDN (CloudFront)                │
│       Global edge caching             │
└──────────────────────────────────────┘

       ┌──────────────┐
       │    Kafka     │
       └──────┬───────┘
              │
       ┌──────▼────────┐
       │Fanout Workers │
       └───────────────┘
\`\`\`

---

## Step 6: Photo Upload Flow

**Flow**:
\`\`\`
1. User uploads photo via mobile app
2. App resizes image to 2 MB max (client-side compression)
3. POST /api/v1/photos with image binary
4. Upload Service:
   a. Validate image (size, format)
   b. Generate unique photo_id (UUID or base62)
   c. Upload original image to S3: s3://photos/{user_id}/{photo_id}_orig.jpg
5. Publish message to Kafka: "new_photo" event
6. Return photo_id to user immediately (async processing)

--- Background Image Processing Worker ---

7. Worker consumes Kafka event
8. Download original from S3
9. Generate multiple sizes:
   - Thumbnail: 150×150 (50 KB)
   - Medium: 1080×1080 (500 KB)
   - Original: As uploaded (2 MB)
10. Upload all sizes to S3
11. Update photos table with image URLs
12. Fanout to followers (if needed)
\`\`\`

**Why Async Processing?**
- Uploading should be fast (< 1 second response)
- Image processing (resize, compression) takes 2-5 seconds
- User doesn't need to wait

**Storage Structure**:
\`\`\`
s3://instagram-photos/
  ├── user_123/
  │   ├── photo_abc_orig.jpg
  │   ├── photo_abc_med.jpg
  │   ├── photo_abc_thumb.jpg
  │   ├── photo_def_orig.jpg
  │   └── ...
  ├── user_456/
  └── ...
\`\`\`

---

## Step 7: Feed Generation (Similar to Twitter)

Instagram uses **hybrid fanout** similar to Twitter:

### Approach: Fanout-on-Write for Regular Users

**Algorithm**:
1. User Bob posts photo
2. If Bob has < 10K followers:
   a. Query: Get Bob's followers
   b. Insert photo_id into each follower's feed table
3. If Bob has >= 10K followers (celebrity):
   - Skip fanout (handle at feed generation time)

**Feed Table**:
\`\`\`sql
CREATE TABLE feed (
    user_id BIGINT NOT NULL,
    photo_id VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, created_at, photo_id),
    INDEX idx_user_created (user_id, created_at)
);
\`\`\`

### Feed Generation Flow

**Flow**:
\`\`\`
1. User requests GET /api/v1/feed?user_id=alice
2. Check Redis cache: GET feed:alice
   → If cache HIT: Return cached feed
3. If cache MISS:
   a. Query feed table (pre-computed from fanout-on-write)
   b. Query photos from celebrities Alice follows (fanout-on-read)
   c. Merge and sort by created_at DESC
   d. Limit to top 100 photos
   e. Cache in Redis (TTL: 10 minutes)
4. Hydrate photo details (batch query):
   - User info (username, profile image)
   - Like counts (from Redis counters)
5. Return feed to user
\`\`\`

**Redis Cache Structure**:
\`\`\`
Key: feed:{user_id}
Value: [photo_id1, photo_id2, ..., photo_id100]
TTL: 600 seconds (10 minutes)
\`\`\`

---

## Step 8: Image Serving via CDN

**Challenge**: Serving billions of images with low latency globally.

**Solution**: Use CloudFront CDN

**Architecture**:
\`\`\`
User Request → CloudFront (Edge Location)
                 ↓ (Cache MISS)
              CloudFront → S3 (Origin)
                 ↓
              CloudFront caches image for 24 hours
                 ↓
              Subsequent requests served from edge (< 50ms)
\`\`\`

**CDN Benefits**:
- **Latency**: < 50ms (vs 200ms from S3 directly)
- **Cost**: CDN cheaper than S3 egress for popular images
- **S3 load**: Reduces S3 GET requests by 90%

**Image URL Structure**:
\`\`\`
https://d1a2b3c4.cloudfront.net/photos/user_123/photo_abc_med.jpg
\`\`\`

### Optimization: Lazy Loading

Mobile app loads:
1. Thumbnail first (50 KB, fast)
2. Medium resolution when user views (500 KB)
3. Original only when user zooms (2 MB)

**Saves bandwidth**: 95% of views only load thumbnail + medium.

---

## Step 9: Database Sharding

### Sharding Strategy

**Photos Table**: Shard by \`user_id\`
- All photos from same user on same shard
- User profile page (common query) hits single shard

**Feed Table**: Shard by \`user_id\`
- Each user's feed on single shard
- Feed generation query hits single shard

**Formula**: \`shard = user_id % NUM_SHARDS\`

### Handling Hot Users (Celebrities)

**Problem**: Celebrity with 100M followers causes hotspot on their shard.

**Solutions**:
1. **Replica reads**: Multiple replicas for hot shards
2. **Caching**: Cache celebrity photos aggressively
3. **Separate shard**: Move celebrities to dedicated shards
4. **Consistent hashing**: Virtual nodes for better distribution

---

## Step 10: Additional Features

### Likes

**Challenge**: Like counts must be updated frequently (thousands/sec per popular photo).

**Solution**: Redis counters + async DB sync

**Flow**:
\`\`\`
1. User likes photo → POST /api/v1/likes
2. Check if already liked (Redis SET: liked:{user_id} contains photo_id)
3. If not liked:
   a. Add to Redis SET: SADD liked:{user_id} photo_id
   b. Increment like count: INCR likes:{photo_id}
   c. Publish to Kafka: "like_event"
4. Background worker syncs Redis → Database hourly
\`\`\`

### Comments

\`\`\`sql
CREATE TABLE comments (
    comment_id BIGINT PRIMARY KEY,
    photo_id VARCHAR(20) NOT NULL,
    user_id BIGINT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_photo_created (photo_id, created_at)
);
\`\`\`

**Pagination**: Use cursor-based pagination to avoid offset issues.

### Search

**Use Elasticsearch**:
- Index users by username
- Index photos by hashtags, captions
- Update index via Kafka consumer

**Query Example**:
\`\`\`
POST /photos/_search
{
  "query": {
    "match": {
      "hashtags": "sunset"
    }
  },
  "sort": [{"created_at": "desc"}],
  "size": 20
}
\`\`\`

### Hashtags

**Popular hashtags table**:
\`\`\`sql
CREATE TABLE hashtag_counts (
    hashtag VARCHAR(100) PRIMARY KEY,
    count INT DEFAULT 0,
    updated_at TIMESTAMP
);
\`\`\`

Track trending hashtags using Redis sorted set (similar to Twitter).

---

## Step 11: Optimizations

### 1. Image Compression

- Use WebP format (30% smaller than JPEG, same quality)
- Progressive JPEG (loads top-to-bottom incrementally)
- Serve different formats based on client: WebP for Chrome, JPEG for Safari

### 2. Pre-signed URLs

Generate temporary S3 URLs for secure upload:
\`\`\`typescript
const url = s3.getSignedUrl('putObject', {
    Bucket: 'instagram-photos',
    Key: \`user_123/photo_abc.jpg\`,
    Expires: 3600  // 1 hour
});
// Return URL to client, client uploads directly to S3
\`\`\`

**Benefits**:
- Reduces server bandwidth (client → S3 direct)
- Faster uploads (no proxy through app server)

### 3. Intelligent Caching

Cache based on photo popularity:
- Hot photos (> 1000 views): Cache for 7 days
- Warm photos (100-1000 views): Cache for 1 day
- Cold photos (< 100 views): No cache or 1 hour

### 4. Photo Deduplication

Some users upload same photo multiple times:
- Compute perceptual hash (pHash) of image
- If duplicate detected, reference existing stored photo
- Saves storage (5-10% savings)

---

## Step 12: Monitoring & Metrics

### Metrics to Track

1. **Upload success rate**: % (should be > 99%)
2. **Image processing time**: p95, p99 (< 5 seconds)
3. **Feed generation latency**: p95 (< 300ms)
4. **CDN cache hit rate**: % (should be > 80%)
5. **S3 storage cost**: $ per TB per month
6. **Database query latency**: p95 (< 50ms)

### Cost Optimization

**Storage costs** (S3):
- Standard: $0.023 per GB per month
- 456 PB = 456,000 TB
- 456,000 × $23 = $10.5M per month
- **Optimization**: Use S3 Intelligent-Tiering
  - Moves old photos to cheaper tiers automatically
  - Reduces cost by 40-70%

---

## Trade-offs

### Consistency vs Availability

**Choice**: Eventual consistency (AP system)
- Feed can be slightly stale (acceptable)
- Like counts approximate (Redis + eventual DB sync)
- Acceptable for social media

### Storage vs Compute

**Multiple image sizes**: Requires 3x storage but saves bandwidth and client-side processing.

### Real-time vs Cost

**Async processing**: Uploads faster for users, but processing delayed 5-10 seconds.

---

## Interview Tips

### What to Clarify

1. **Scale**: How many users? Photos per day?
2. **Media types**: Photos only or videos too?
3. **Features**: Stories? Direct messages?
4. **Latency**: Acceptable feed load time?

### What to Emphasize

1. **CDN for image delivery**: Critical for global low latency
2. **Hybrid fanout**: Similar to Twitter but with photos
3. **Async image processing**: Don't block upload
4. **Sharding by user_id**: Optimizes common queries

### Common Mistakes

1. ❌ Storing images in database (use object storage)
2. ❌ Not using CDN (expensive and slow)
3. ❌ Synchronous image processing (blocks upload)
4. ❌ Ignoring storage costs at petabyte scale

### Follow-up Questions

- "How would you implement Stories (24-hour expiration)?"
- "How do you handle video uploads (100 MB files)?"
- "How would you build the Explore page (personalized recommendations)?"
- "What if users upload copyrighted images?"

---

## Summary

**Core Components**:
1. **Upload Service**: Handle photo uploads, validation
2. **Image Processing**: Async resize, compression
3. **Feed Service**: Generate personalized feeds (hybrid fanout)
4. **S3**: Store 456 PB of photos
5. **CDN**: Serve images with < 50ms latency globally
6. **Redis**: Cache feeds (10 TB) and like counters
7. **Database (Sharded)**: Metadata, users, follows

**Key Decisions**:
- ✅ Store images in S3, serve via CloudFront CDN
- ✅ Async image processing (3 sizes: thumb, med, orig)
- ✅ Hybrid fanout for feed generation
- ✅ Redis counters for likes (sync to DB hourly)
- ✅ Shard by user_id for query locality
- ✅ Eventual consistency acceptable

**Capacity**:
- 100M photos/day (1,160 uploads/sec)
- 1B feed views/day (11,500 views/sec)
- 456 PB storage over 5 years
- 10 TB Redis cache

This design handles **Instagram-scale** with **sub-300ms feed loads** and **global image delivery** via CDN, while optimizing storage costs through intelligent tiering and caching.`,
};
