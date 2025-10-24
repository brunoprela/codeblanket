/**
 * Design TinyURL (URL Shortener) Section
 */

export const tinyurlSection = {
  id: 'tinyurl',
  title: 'Design TinyURL (URL Shortener)',
  content: `A URL shortening service like TinyURL, bit.ly, or goo.gl converts long URLs into short, manageable links. This is one of the most popular system design interview questions because it covers fundamental concepts: encoding, database design, scalability, and caching.

## Problem Statement

Design a URL shortening service that:
- Takes a long URL and generates a unique short URL
- Redirects users from short URL to original long URL
- Provides analytics (optional): click count, timestamp, location
- Handles massive scale: billions of URLs, high read traffic

**Example**:
\`\`\`
Input:  https://www.example.com/very/long/path/to/article?param1=value1&param2=value2
Output: https://tiny.url/abc123

When user visits https://tiny.url/abc123 → Redirects to original URL
\`\`\`

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **URL Shortening**: Given long URL, return unique short URL
2. **URL Redirection**: Given short URL, redirect to original URL
3. **Custom URLs** (optional): Users can specify custom short URL
4. **Expiration**: URLs can have TTL (optional)
5. **Analytics**: Track clicks, referrer, location (optional)

### Non-Functional Requirements

1. **High Availability**: System must be always available (reads are critical)
2. **Low Latency**: Redirection should be fast (<100ms)
3. **Scalable**: Handle billions of URLs and millions of requests/sec
4. **Durable**: URLs must not be lost once created
5. **Unpredictable**: Short URLs should not be easily guessable

### Out of Scope

- User authentication/accounts
- URL validation/malware checking
- Rate limiting
- API keys

---

## Step 2: Capacity Estimation

### Traffic Estimation

**Assumptions**:
- 500 million new URLs generated per month
- Read:Write ratio = 100:1 (URL redirections >> creations)

**Writes (URL creation)**:
- 500M URLs/month = ~200 URLs/sec
- Peak: 400 URLs/sec (2x average)

**Reads (URL redirection)**:
- 100:1 ratio → 50B redirections/month
- 50B/month = ~20,000 requests/sec
- Peak: 40,000 requests/sec

### Storage Estimation

**URLs stored**:
- 500M new URLs/month
- Keep for 10 years
- Total: 500M × 12 months × 10 years = 60 billion URLs

**Storage per URL**:
- Short URL key: 7 bytes
- Long URL: 500 bytes (average)
- Metadata: 100 bytes (created_at, expiry, user_id)
- Total: ~600 bytes per URL

**Total storage**:
- 60B URLs × 600 bytes = 36 TB over 10 years

### Bandwidth Estimation

**Write bandwidth**:
- 200 URLs/sec × 600 bytes = 120 KB/sec (~1 Mbps)

**Read bandwidth**:
- 20,000 requests/sec × 500 bytes = 10 MB/sec (~80 Mbps)

### Cache Memory

Cache 20% of hot URLs (80/20 rule):
- 20K requests/sec × 86,400 sec/day = 1.7B requests/day
- Assume 20% unique URLs cached
- 1.7B × 0.20 = 340M URLs cached
- 340M × 500 bytes = 170 GB cache

---

## Step 3: System API Design

### REST API

**Shorten URL**:
\`\`\`
POST /api/v1/shorten
Content-Type: application/json

Request Body:
{
  "long_url": "https://example.com/very/long/url",
  "custom_alias": "my-custom-url",  // optional
  "expiry_date": "2025-12-31"       // optional
}

Response (201 Created):
{
  "short_url": "https://tiny.url/abc123",
  "long_url": "https://example.com/very/long/url",
  "created_at": "2024-10-24T10:00:00Z",
  "expiry_date": "2025-12-31"
}
\`\`\`

**Redirect (handled by HTTP 301/302)**:
\`\`\`
GET /abc123

Response:
HTTP 301 Moved Permanently
Location: https://example.com/very/long/url
\`\`\`

**Get Analytics** (optional):
\`\`\`
GET /api/v1/analytics/abc123

Response:
{
  "short_url": "abc123",
  "total_clicks": 1523,
  "created_at": "2024-10-24T10:00:00Z",
  "last_accessed": "2024-10-25T15:30:00Z"
}
\`\`\`

---

## Step 4: Database Schema

### URLs Table

\`\`\`sql
CREATE TABLE urls (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_url VARCHAR(10) UNIQUE NOT NULL,
    long_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry_date TIMESTAMP NULL,
    user_id BIGINT,
    INDEX idx_short_url (short_url),
    INDEX idx_expiry_date (expiry_date)
);
\`\`\`

### Analytics Table (Optional)

\`\`\`sql
CREATE TABLE analytics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_url VARCHAR(10) NOT NULL,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    referrer VARCHAR(500),
    user_agent VARCHAR(500),
    ip_address VARCHAR(45),
    INDEX idx_short_url_time (short_url, accessed_at)
);
\`\`\`

---

## Step 5: Short URL Generation

This is the **core algorithmic challenge**. We need to generate unique 6-7 character keys from billions of URLs.

### Approach 1: Hash + Collision Handling

**Algorithm**:
1. Hash long URL using MD5/SHA256
2. Take first 6-8 characters (base64)
3. Check if key exists in database
4. If collision, append counter/try different substring

**Pros**:
- Same URL always generates same hash (idempotent)
- Fast computation

**Cons**:
- Collision handling adds complexity
- Extra database queries on collision
- Not truly unique without checking DB

### Approach 2: Base62 Encoding (Recommended)

**Algorithm**:
1. Use auto-incrementing database ID
2. Encode ID to base62 (a-z, A-Z, 0-9)
3. Result: unique short key

**Base62 encoding**:
\`\`\`
Characters: [a-z][A-Z][0-9] = 62 characters
6 characters: 62^6 = 56 billion unique URLs
7 characters: 62^7 = 3.5 trillion unique URLs
\`\`\`

**Implementation**:
\`\`\`typescript
function encodeBase62(id: number): string {
    const chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let encoded = '';
    
    while (id > 0) {
        encoded = chars[id % 62] + encoded;
        id = Math.floor(id / 62);
    }
    
    return encoded || '0';
}

// Example:
// ID 125 → "21" in base62
// ID 12345 → "3D7" in base62
\`\`\`

**Pros**:
- ✅ Guaranteed unique (based on DB ID)
- ✅ No collision handling needed
- ✅ Simple and predictable

**Cons**:
- ❌ Sequential IDs are predictable (security concern)
- ❌ Requires database write to get ID

### Approach 3: Random Generation + Uniqueness Check

**Algorithm**:
1. Generate random 6-7 character string
2. Check if exists in database
3. If collision, regenerate
4. Insert into database

**Pros**:
- Unpredictable URLs
- No sequential pattern

**Cons**:
- Collision probability increases over time
- Multiple database queries on collision

### Recommended: Base62 with Random Offset

**Hybrid approach**:
1. Generate base62 from (ID + random_offset)
2. This breaks sequential pattern while maintaining uniqueness

\`\`\`typescript
const RANDOM_OFFSET = generateRandomOffset(); // e.g., 1000000000

function generateShortURL(id: number): string {
    return encodeBase62(id + RANDOM_OFFSET);
}
\`\`\`

---

## Step 6: High-Level Architecture

\`\`\`
                                    ┌─────────────────┐
                                    │   Load Balancer │
                                    └────────┬────────┘
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
                     ▼                       ▼                       ▼
              ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
              │ App Server  │         │ App Server  │         │ App Server  │
              │  (Write)    │         │   (Read)    │         │   (Read)    │
              └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
                     │                       │                       │
                     │                       ├───────────────────────┘
                     │                       │
                     │                       ▼
                     │              ┌─────────────────┐
                     │              │  Redis Cache    │
                     │              │  (LRU, 170 GB)  │
                     │              └─────────────────┘
                     │                       │
                     │                       │ Cache Miss
                     ▼                       ▼
              ┌──────────────────────────────────────┐
              │       Primary SQL Database           │
              │    (URLs, Analytics)                 │
              └──────────────┬───────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────────┐
              │      Read Replicas (3-5 replicas)    │
              │      (Handle read traffic)           │
              └──────────────────────────────────────┘
\`\`\`

### Component Breakdown

**Load Balancer**:
- Distributes traffic across app servers
- Health checks
- SSL termination

**Application Servers**:
- Separate read (redirect) and write (create) servers
- Read servers: stateless, horizontally scalable
- Write servers: handle URL creation

**Cache Layer (Redis)**:
- Cache hot URLs (20% = 80% traffic)
- TTL: 24 hours
- Cache-aside pattern
- LRU eviction policy

**Database**:
- Primary: Handles all writes
- Replicas: Handle read traffic (redirects)
- Sharding strategy (if needed): Hash-based on short_url

---

## Step 7: URL Creation Flow

**Flow**:
\`\`\`
1. Client sends POST /api/v1/shorten with long_url
2. App server checks if long_url already exists (optional optimization)
3. Insert into database → Get auto-increment ID
4. Encode ID to base62 → short_url
5. Update row with short_url
6. Return short_url to client
\`\`\`

**Optimization**: Two-phase approach
\`\`\`sql
-- Transaction:
BEGIN;
INSERT INTO urls (long_url, created_at) VALUES (?, NOW());
SET @id = LAST_INSERT_ID();
UPDATE urls SET short_url = ENCODE_BASE62(@id) WHERE id = @id;
COMMIT;
\`\`\`

---

## Step 8: URL Redirection Flow

**Flow**:
\`\`\`
1. Client requests GET /abc123
2. Check Redis cache for "abc123"
   → Cache HIT: Return long_url, send 301 redirect
   → Cache MISS: Query database
3. If found in DB:
   - Store in Redis with TTL
   - Return 301/302 redirect
4. If not found: Return 404
5. Async: Log analytics (click count, timestamp)
\`\`\`

### 301 vs 302 Redirect

**301 (Permanent)**:
- Browser caches redirect
- Subsequent requests don't hit server
- Cannot track analytics after first visit

**302 (Temporary)**:
- Browser doesn't cache
- Every request hits server
- Can track all analytics

**Recommendation**: Use **302** for analytics, **301** if analytics not needed.

---

## Step 9: Database Sharding

When single database can't handle load, shard by short_url.

### Sharding Strategy: Hash-Based

\`\`\`
Shard = hash(short_url) % NUM_SHARDS

Example (4 shards):
- "abc123" → hash % 4 = Shard 2
- "xyz789" → hash % 4 = Shard 1
\`\`\`

**Pros**:
- Even distribution
- Simple lookup

**Cons**:
- Adding shards requires rehashing (use consistent hashing)

---

## Step 10: Additional Considerations

### Custom URLs

Allow users to choose custom short URLs:
\`\`\`
POST /api/v1/shorten
{
  "long_url": "https://example.com",
  "custom_alias": "my-link"
}

Result: https://tiny.url/my-link
\`\`\`

**Challenge**: Check uniqueness before insertion
**Solution**: UNIQUE constraint on short_url column

### URL Expiration

**Cleanup strategy**:
1. **Lazy deletion**: Check expiry_date on read
2. **Active cleanup**: Background job deletes expired URLs
3. **Archive**: Move expired URLs to archive table

### Rate Limiting

Prevent abuse:
- Limit: 10 URLs per hour per IP
- Use Redis with sliding window counter

### Analytics at Scale

**Problem**: Writing to analytics table for every redirect is expensive.

**Solutions**:
1. **Async writes**: Queue analytics events (Kafka)
2. **Batch inserts**: Accumulate and batch write every 10 seconds
3. **Sampling**: Only log 10% of redirects for trending analysis
4. **Separate service**: Dedicated analytics service

### Security

1. **Prevent abuse**: Rate limiting
2. **Malware/phishing**: Scan long URLs before shortening
3. **HTTPS**: Enforce SSL
4. **No sequential IDs**: Use random offset or hashing

---

## Trade-offs and Optimizations

### Consistency vs Availability

**Scenario**: Primary database goes down

**Option 1**: Fail writes, continue serving reads from replicas (AP)
**Option 2**: Reject all requests until primary is back (CP)

**Recommendation**: **AP** - Reads are 100x more critical than writes

### Caching Strategy

**Cache-Aside** (Recommended):
\`\`\`
1. Check cache
2. If miss, query database
3. Store in cache before returning
\`\`\`

**Cache-Through**:
- All reads/writes go through cache
- Cache syncs with database
- More complex

### Database Choice

**SQL (PostgreSQL/MySQL)**: ✅ Recommended
- Strong consistency
- ACID transactions
- Simple schema

**NoSQL (DynamoDB/Cassandra)**:
- Better for massive scale (trillions of URLs)
- Eventual consistency acceptable
- More operational complexity

---

## Interview Tips

### What to Clarify

1. **Scale**: How many URLs? Request rate?
2. **Analytics**: Do we need click tracking?
3. **Custom URLs**: Can users choose their short URL?
4. **Expiration**: Do URLs expire?
5. **API only or Web UI**: Design scope

### What to Emphasize

1. **Base62 encoding**: Explain clearly
2. **Caching**: Show understanding of cache layers
3. **Database sharding**: If scale is massive
4. **Trade-offs**: 301 vs 302, SQL vs NoSQL

### Common Mistakes

1. ❌ Jumping to distributed systems without justifying scale
2. ❌ Over-engineering (don't need Kafka for 200 writes/sec)
3. ❌ Not handling collisions in hash-based approach
4. ❌ Ignoring analytics impact on write throughput

### Follow-up Questions

- "How would you handle 1 trillion URLs?"
- "What if a celebrity shares a URL and traffic spikes 100x?"
- "How do you prevent malicious URLs?"
- "How would you implement custom domains? (e.g., brand.ly/abc123)"

---

## Summary

**Core Components**:
1. **API Layer**: REST endpoints for create/redirect
2. **Application Servers**: Stateless, horizontally scalable
3. **Cache**: Redis for hot URLs (170 GB for 20% traffic)
4. **Database**: SQL with replication (primary + replicas)
5. **Short URL Generation**: Base62 encoding of auto-increment ID

**Key Design Decisions**:
- ✅ Base62 encoding for guaranteed uniqueness
- ✅ 302 redirects for analytics tracking
- ✅ Cache-aside pattern with Redis
- ✅ Separate read and write paths
- ✅ Async analytics processing

**Capacity**:
- 60 billion URLs over 10 years
- 20,000 reads/sec, 200 writes/sec
- 36 TB storage, 170 GB cache

This design handles **millions of requests per second** with **sub-100ms latency** while remaining simple and cost-effective.`,
};
