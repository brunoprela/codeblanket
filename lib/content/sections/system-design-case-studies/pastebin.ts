/**
 * Design Pastebin Section
 */

export const pastebinSection = {
  id: 'pastebin',
  title: 'Design Pastebin',
  content: `Pastebin is a web application where users can store plain text for sharing. Services like Pastebin.com, GitHub Gist, and PasteBin allow developers to share code snippets, logs, configuration files, and text data with expiration and syntax highlighting.

## Problem Statement

Design a Pastebin-like service that allows users to:
- **Upload/paste** text content
- **Generate** unique URL for sharing
- **View** pasted content
- **Set expiration** (1 hour, 1 day, 1 week, never)
- **Support custom URLs** (optional)
- **Syntax highlighting** for code (optional)
- **Private/public** pastes

**Example**:
\`\`\`
User pastes: "console.log('Hello World');"
System generates: https://pastebin.com/a3Bk9xZ1
Anyone with link can view the content
Paste expires after 24 hours (if configured)
\`\`\`

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Create Paste**: Users can paste text and get unique URL
2. **Read Paste**: Users can view paste via URL
3. **Expiration**: Pastes can expire (1 hour, 1 day, 1 week, never)
4. **Custom URL**: Users can choose custom URL alias (optional)
5. **Private vs Public**: Private pastes require authentication
6. **Syntax Highlighting**: Display code with proper formatting (optional)
7. **Raw View**: View plain text without formatting

### Non-Functional Requirements

1. **High Availability**: 99.9% uptime
2. **Low Latency**: < 100ms to retrieve paste
3. **Scalable**: Handle millions of pastes per day
4. **Durable**: Pastes must not be lost
5. **Secure**: Private pastes only accessible by creator
6. **Size Limit**: Max 10 MB per paste

### Out of Scope

- User accounts (simple version)
- Editing pastes after creation
- Versioning
- Collaborative editing
- Comments

---

## Step 2: Capacity Estimation

### Traffic Estimation

**Assumptions**:
- 1 million new pastes per day
- Read:Write ratio = 10:1 (views >> creations)

**Writes (Create paste)**:
- 1M pastes/day = ~12 pastes/sec
- Peak: 25 pastes/sec (2x average)

**Reads (View paste)**:
- 10:1 ratio → 10M views/day
- 10M/day = ~120 reads/sec
- Peak: 240 reads/sec

### Storage Estimation

**Paste retention**:
- 30% expire in 1 day (deleted)
- 20% expire in 1 week
- 50% never expire
- Average retention: 5 years

**Pastes stored**:
- 1M pastes/day
- With 50% never expiring + others with retention
- Effective storage: ~500K pastes/day accumulating
- Over 5 years: 500K × 365 × 5 = 912 million pastes

**Storage per paste**:
- Average paste size: 10 KB (code snippets, logs)
- Metadata: 1 KB (URL, expiry, created_at, user)
- Total: ~11 KB per paste

**Total storage**:
- 912M pastes × 11 KB = 10 TB over 5 years

### Bandwidth Estimation

**Write bandwidth**:
- 12 pastes/sec × 11 KB = 132 KB/sec (~1 Mbps)

**Read bandwidth**:
- 120 reads/sec × 11 KB = 1.3 MB/sec (~10 Mbps)

### Cache Memory

Cache 20% of hot pastes:
- 120 reads/sec × 86,400 sec/day = 10.3M reads/day
- Assume 20% unique pastes cached
- 10.3M × 0.20 = 2M pastes cached
- 2M × 11 KB = 22 GB cache

---

## Step 3: System API Design

### REST API

**Create Paste**:
\`\`\`
POST /api/v1/paste
Content-Type: application/json

Request:
{
  "content": "console.log('Hello World');",
  "title": "JavaScript Hello World",
  "syntax": "javascript",
  "expiration": "1d",          // "1h", "1d", "1w", "never"
  "custom_url": "my-js-code",  // optional
  "private": false
}

Response (201 Created):
{
  "paste_id": "a3Bk9xZ1",
  "url": "https://pastebin.com/a3Bk9xZ1",
  "raw_url": "https://pastebin.com/raw/a3Bk9xZ1",
  "created_at": "2024-10-24T10:00:00Z",
  "expires_at": "2024-10-25T10:00:00Z"
}
\`\`\`

**Get Paste**:
\`\`\`
GET /api/v1/paste/{paste_id}

Response (200 OK):
{
  "paste_id": "a3Bk9xZ1",
  "content": "console.log('Hello World');",
  "title": "JavaScript Hello World",
  "syntax": "javascript",
  "created_at": "2024-10-24T10:00:00Z",
  "expires_at": "2024-10-25T10:00:00Z",
  "views": 142
}
\`\`\`

**Get Raw Paste**:
\`\`\`
GET /raw/{paste_id}

Response (200 OK, text/plain):
console.log('Hello World');
\`\`\`

---

## Step 4: Database Schema

### Pastes Table

\`\`\`sql
CREATE TABLE pastes (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    paste_id VARCHAR(10) UNIQUE NOT NULL,
    title VARCHAR(255),
    content MEDIUMTEXT NOT NULL,          -- Up to 16 MB
    syntax VARCHAR(50),                    -- "javascript", "python", etc.
    is_private BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    user_id BIGINT NULL,                   -- For authenticated users
    view_count INT DEFAULT 0,
    content_hash VARCHAR(64),              -- SHA-256 for deduplication
    INDEX idx_paste_id (paste_id),
    INDEX idx_expires_at (expires_at),
    INDEX idx_content_hash (content_hash)
);
\`\`\`

### Why MEDIUMTEXT?

- **TEXT**: Up to 64 KB
- **MEDIUMTEXT**: Up to 16 MB (sufficient for most pastes)
- **LONGTEXT**: Up to 4 GB (overkill, use object storage instead)

For pastes > 10 MB, store in S3 and keep only metadata in database.

---

## Step 5: Paste ID Generation

Similar to URL shortener, we need unique short IDs.

### Approach: Base62 Encoding (Recommended)

\`\`\`
6 characters: 62^6 = 56 billion pastes
7 characters: 62^7 = 3.5 trillion pastes

For 912 million pastes, 6 characters is sufficient.
\`\`\`

**Implementation**:
\`\`\`typescript
function generatePasteID(id: number): string {
    const chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let encoded = '';
    
    while (id > 0) {
        encoded = chars[id % 62] + encoded;
        id = Math.floor(id / 62);
    }
    
    return encoded.padStart(6, '0');  // Minimum 6 characters
}
\`\`\`

**Flow**:
1. Insert paste into database → Get auto-increment ID
2. Encode ID to base62 → paste_id
3. Update row with paste_id

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
              └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
                     │                       │                       │
                     │                       ├───────────────────────┘
                     │                       │
                     │                       ▼
                     │              ┌─────────────────┐
                     │              │  Redis Cache    │
                     │              │  (22 GB)        │
                     │              └─────────────────┘
                     │                       │
                     │                       │ Cache Miss
                     ▼                       ▼
              ┌──────────────────────────────────────┐
              │       Primary SQL Database           │
              │    (Pastes Table)                    │
              └──────────────┬───────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────────┐
              │      Read Replicas (3 replicas)      │
              │      (Handle read traffic)           │
              └──────────────────────────────────────┘

              ┌──────────────────────────────────────┐
              │       Object Storage (S3)            │
              │  (For large pastes > 10 MB)          │
              └──────────────────────────────────────┘
\`\`\`

---

## Step 7: Create Paste Flow

**Flow**:
\`\`\`
1. Client sends POST /api/v1/paste with content
2. App server validates content size (< 10 MB)
3. Check content_hash for deduplication (optional)
   → If exists and not expired, return existing paste_id
4. If content > 10 MB:
   → Upload to S3
   → Store S3 key in database instead of content
5. Insert into database → Get auto-increment ID
6. Generate paste_id = encodeBase62(ID)
7. Update row with paste_id
8. Calculate expires_at timestamp based on expiration setting
9. Return paste_id and URL to client
\`\`\`

### Deduplication (Optional Optimization)

Many users paste identical content (common error messages, code snippets).

**Strategy**:
1. Compute SHA-256 hash of content
2. Check if hash exists in database
3. If exists and not expired → Return existing paste_id
4. Else → Create new paste

**Storage savings**: 20-30% for duplicate content.

---

## Step 8: View Paste Flow

**Flow**:
\`\`\`
1. Client requests GET /a3Bk9xZ1
2. Check Redis cache for paste_id
   → Cache HIT: Return content
   → Cache MISS: Query database
3. Check if paste exists and not expired
4. If content stored in S3:
   → Fetch from S3 (with CDN)
5. Store in cache with TTL
6. Increment view_count (async)
7. Return paste content with syntax highlighting
\`\`\`

### Handling Expiration

**Option 1: Lazy Deletion**
- Check \`expires_at\` during read
- If expired, return 404
- Background job deletes expired pastes every 24 hours

**Option 2: Active Deletion**
- Background job runs every hour
- Deletes pastes where \`expires_at < NOW()\`
- More storage efficient

**Recommendation**: Lazy deletion during reads + daily cleanup job.

---

## Step 9: Caching Strategy

### Cache-Aside Pattern

\`\`\`typescript
async function getPaste(pasteId: string): Paste {
    // 1. Check cache
    const cached = await redis.get(\`paste:\${pasteId}\`);
    if (cached) {
        return JSON.parse(cached);
    }
    
    // 2. Query database
    const paste = await db.query(
        'SELECT * FROM pastes WHERE paste_id = ? AND (expires_at IS NULL OR expires_at > NOW())',
        [pasteId]
    );
    
    if (!paste) {
        // Cache 404 to prevent repeated DB queries
        await redis.set(\`paste:\${pasteId}\`, 'NOT_FOUND', 'EX', 3600);
        return null;
    }
    
    // 3. If large paste, fetch from S3
    if (paste.s3_key) {
        paste.content = await s3.getObject(paste.s3_key);
    }
    
    // 4. Store in cache
    await redis.set(\`paste:\${pasteId}\`, JSON.stringify(paste), 'EX', 86400);  // 24 hours
    
    return paste;
}
\`\`\`

### Cache Keys

- \`paste:{paste_id}\` → Full paste object
- TTL: 24 hours (or until \`expires_at\`, whichever is earlier)
- Cache negative results (404s) for 1 hour

---

## Step 10: Optimizations

### 1. Large Pastes → Object Storage

For pastes > 10 MB:
\`\`\`
1. Upload content to S3
2. Store S3 key in database: s3://bucket/pastes/a3Bk9xZ1.txt
3. Serve via CloudFront CDN for fast access
4. Database only stores metadata (< 1 KB per paste)
\`\`\`

**Schema modification**:
\`\`\`sql
ALTER TABLE pastes ADD COLUMN s3_key VARCHAR(255) NULL;

-- If s3_key is NULL, content is in 'content' column
-- If s3_key is set, content is in S3
\`\`\`

### 2. Content Compression

Store content compressed in database:
\`\`\`typescript
// Before insert
const compressed = gzip(content);
paste.content = compressed;

// Before return
const decompressed = gunzip(paste.content);
return decompressed;
\`\`\`

**Compression ratio**: 5:1 for code (JSON, XML, logs)
**Trade-off**: CPU for storage savings and network bandwidth

### 3. Syntax Highlighting

**Client-side** (Recommended):
- Use highlight.js or Prism.js in browser
- Server sends raw content
- Client renders with syntax highlighting
- Reduces server load

**Server-side**:
- Pre-render HTML with syntax highlighting
- Cache rendered HTML
- Faster for users, more server load

### 4. Private Pastes

**Authentication**:
\`\`\`sql
ALTER TABLE pastes ADD COLUMN access_token VARCHAR(64) NULL;

-- Generate random token for private pastes
-- Share URL: https://pastebin.com/a3Bk9xZ1?token=xyz123...
\`\`\`

**Access control**:
1. If paste is private, require token in URL
2. Validate token before serving content
3. Do not cache private pastes in shared Redis

---

## Step 11: Additional Considerations

### Custom URLs

Allow users to specify custom paste_id:
\`\`\`
POST /api/v1/paste
{
  "content": "...",
  "custom_url": "my-code"
}

Result: https://pastebin.com/my-code
\`\`\`

**Implementation**:
- Check if custom_url already exists (UNIQUE constraint)
- If available, use custom_url as paste_id
- If taken, return error

### Rate Limiting

Prevent spam:
- Limit: 10 pastes per hour per IP
- Use Redis with sliding window counter
- Block IPs after repeated violations

### View Counter

**Problem**: Incrementing view_count on every read is expensive.

**Solutions**:
1. **Async update**: Queue increment events, batch update every 10 seconds
2. **Approximate counting**: Use Redis counter, sync to DB hourly
3. **Skip for cached requests**: Only count cache misses

**Recommended**: Redis counter + hourly DB sync
\`\`\`typescript
// On read
await redis.incr(\`views:\${pasteId}\`);

// Background job every hour
async function syncViewCounts() {
    const keys = await redis.keys('views:*');
    for (const key of keys) {
        const pasteId = key.split(':')[1];
        const count = await redis.get(key);
        await db.query(
            'UPDATE pastes SET view_count = view_count + ? WHERE paste_id = ?',
            [count, pasteId]
        );
        await redis.del(key);
    }
}
\`\`\`

### Malicious Content

**Challenges**:
- Malware links
- Spam
- Copyright violations

**Solutions**:
1. Content scanning with VirusTotal API
2. Blacklist known malicious domains
3. User reporting mechanism
4. Automated takedown for DMCA claims

---

## Step 12: Database Sharding (If Needed)

### When to Shard

- Storage exceeds single DB capacity (> 10 TB)
- Write load exceeds single DB capacity (> 5K writes/sec)

For Pastebin (12 writes/sec, 120 reads/sec), **single DB + replicas is sufficient**.

### Sharding Strategy (If Needed)

**Hash-based sharding by paste_id**:
\`\`\`
Shard = hash(paste_id) % NUM_SHARDS

Example (4 shards):
- "a3Bk9x" → hash % 4 = Shard 2
- "xYz789" → hash % 4 = Shard 1
\`\`\`

**Lookup**:
1. Hash paste_id
2. Route to correct shard
3. Query shard database

---

## Step 13: Monitoring & Observability

### Metrics to Track

1. **Paste creation rate**: pastes/sec
2. **Paste read rate**: reads/sec
3. **Cache hit rate**: % (should be > 70%)
4. **Expiration deletions**: expired pastes/day
5. **Average paste size**: KB
6. **Storage usage**: TB used
7. **Error rate**: 4xx, 5xx responses
8. **Latency**: p50, p95, p99 for reads

### Alerts

- Cache hit rate < 60% → Increase cache size
- Error rate > 1% → Investigate failed requests
- Storage > 80% capacity → Add storage
- Latency p99 > 500ms → Scale read replicas

---

## Trade-offs and Decisions

### SQL vs NoSQL

**SQL (PostgreSQL/MySQL)**: ✅ Recommended
- Simple schema
- ACID transactions
- Easy queries (expiration, user_id filtering)

**NoSQL (DynamoDB/Cassandra)**:
- Better for massive scale (billions of pastes)
- Eventual consistency acceptable
- More complex expiration handling

### Caching: Redis vs CDN

**Redis**: For dynamic content, frequently accessed pastes
**CDN (CloudFront)**: For static large files in S3

**Hybrid**: Cache metadata in Redis, serve large content from CDN.

### Client-side vs Server-side Syntax Highlighting

**Client-side** (Recommended):
- Offloads CPU to client
- Simpler server architecture
- Works offline once loaded

**Server-side**:
- Faster initial render
- Better for SEO (if public pastes)
- More server CPU load

---

## Interview Tips

### What to Clarify

1. **Scale**: How many pastes per day?
2. **Size limit**: What's the max paste size?
3. **Expiration**: Do all pastes expire?
4. **Features**: Private pastes? Custom URLs? Syntax highlighting?
5. **Authentication**: Do users have accounts?

### What to Emphasize

1. **Base62 encoding** for paste IDs
2. **Caching strategy** to reduce DB load
3. **Object storage** for large pastes
4. **Lazy deletion** for expired pastes

### Common Mistakes

1. ❌ Not handling expiration (expired pastes take space)
2. ❌ Storing all content in database (use S3 for large files)
3. ❌ Updating view_count synchronously (kills write performance)
4. ❌ Over-engineering with sharding when single DB is sufficient

### Follow-up Questions

- "How would you implement paste editing?"
- "How do you handle 100 MB paste files?"
- "How would you add user accounts and paste history?"
- "How do you prevent abuse and spam?"

---

## Summary

**Core Components**:
1. **API Layer**: Create, read, raw endpoints
2. **Application Servers**: Stateless, horizontally scalable
3. **Cache (Redis)**: 22 GB for hot pastes
4. **Database**: SQL with replicas (10 TB storage)
5. **Object Storage (S3)**: For large pastes > 10 MB
6. **CDN**: Serve static content and large files

**Key Design Decisions**:
- ✅ Base62 encoding for guaranteed uniqueness
- ✅ Cache-aside pattern with 24-hour TTL
- ✅ Store large pastes in S3, serve via CDN
- ✅ Lazy deletion + daily cleanup for expired pastes
- ✅ Client-side syntax highlighting
- ✅ Async view counter updates

**Capacity**:
- 912 million pastes over 5 years
- 120 reads/sec, 12 writes/sec
- 10 TB storage, 22 GB cache

This design handles **millions of pastes per day** with **sub-100ms latency** while optimizing for storage and cost efficiency.`,
};
