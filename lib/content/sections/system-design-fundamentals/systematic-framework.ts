/**
 * Systematic Problem-Solving Framework Section
 */

export const systematicframeworkSection = {
  id: 'systematic-framework',
  title: 'Systematic Problem-Solving Framework',
  content: `A systematic approach to system design interviews helps you stay organized, cover all important aspects, and demonstrate structured thinking. This framework works for any design problem.

## The 4-Step Framework

### **Step 1: Requirements & Scope (5-10 minutes)**
### **Step 2: High-Level Design (10-15 minutes)**
### **Step 3: Deep Dive (20-25 minutes)**
### **Step 4: Wrap Up & Optimizations (5 minutes)**

Total: 45-60 minutes

---

## Step 1: Requirements & Scope (5-10 min)

### **Goal:** Clarify what you're building and for whom.

### **A. Clarify Functional Requirements**

**Ask:** What features must the system support?

**Example (Twitter):**
✅ "Should we support:"
- Post tweets (280 chars)?
- Follow/unfollow users?
- View personalized timeline?
- Like and retweet?
- Direct messaging?
- Notifications?
- Trending topics?

**Prioritize:** Core features vs nice-to-have
- Core: Post, follow, timeline
- Nice-to-have: Trending, verified badges

### **B. Clarify Non-Functional Requirements**

**Ask about scale:**
- How many daily active users?
- How many tweets per day?
- How many followers per user (average/max)?
- Read vs write ratio?

**Ask about performance:**
- What\'s acceptable latency for timeline loading?
- Real-time updates or eventual consistency acceptable?

**Ask about availability:**
- What's the uptime requirement? 99.9%? 99.99%?
- Can we tolerate brief inconsistencies?

### **C. State Assumptions**

Write them down explicitly:
- "Assuming 300M DAU"
- "500M tweets/day"
- "100M average read/write ratio"
- "Eventual consistency acceptable"
- "99.9% availability target"

### **D. Define Scope Boundaries**

What you WON'T cover:
- "I'll focus on core tweet posting and timeline viewing"
- "I'll skip analytics, ads, and recommendation algorithms"

### **Time Check:** 5-10 minutes spent

---

## Step 2: High-Level Design (10-15 min)

### **Goal:** Create the overall architecture that everyone can understand.

### **A. Calculate Back-of-Envelope Estimations**

**Storage:**
- 500M tweets/day × 280 bytes = 140 GB/day (text)
- With media: ~50 TB/day
- 5 years: ~90 PB

**Traffic:**
- Writes: 500M/day ÷ 86,400 = ~6K tweets/sec
- Reads (100×): 600K reads/sec
- Peak (3×): 1.8M reads/sec

**Conclusion:** Need distributed system, caching, CDN

### **B. Define Core Components**

**Draw boxes for:**
1. **Client** (web/mobile apps)
2. **Load Balancer** (distribute traffic)
3. **API Servers** (business logic)
4. **Database** (persistent storage)
5. **Cache** (Redis for performance)
6. **CDN** (media delivery)
7. **Message Queue** (async processing)

### **C. Define APIs**

**Key endpoints:**
\`\`\`
    POST / api / v1 / tweets
    Request: { user_id, content, media_urls[] }
    Response: { tweet_id, timestamp }

    GET / api / v1 / timeline /: user_id
    Response: { tweets: [{ tweet_id, content, author, timestamp }] }

    POST / api / v1 / follow
    Request: { follower_id, followee_id }
    Response: { success: true }
    \`\`\`

### **D. Choose Database Schema**

**Users table:**
- user_id (PK)
- username
- email
- created_at

**Tweets table:**
- tweet_id (PK)
- user_id (FK)
- content
- created_at

**Follows table:**
- follower_id (FK)
- followee_id (FK)
- created_at

### **E. Draw the High-Level Diagram**

\`\`\`
    [Clients] →[Load Balancer] →[API Servers] →[Cache(Redis)]
                                             ↓
    [Database]
                                             ↓
    [Message Queue]
                                             ↓
    [Background Workers]
        \`\`\`

### **F. Explain the Flow**

**Write flow:**
1. User posts tweet via mobile app
2. Load balancer routes to API server
3. API server validates and writes to database
4. Publish event to message queue
5. Background workers update followers' timelines (fanout)
6. Return success to user

**Read flow:**
1. User requests timeline
2. API server checks cache
3. If cache hit: return immediately
4. If cache miss: query database, populate cache
5. Return timeline to user

### **Time Check:** 15-25 minutes total

---

## Step 3: Deep Dive (20-25 min)

### **Goal:** Dig deep into 2-3 critical components.

### **What to Deep Dive Into?**

**Let interviewer guide, or choose based on:**
- Most complex components
- Bottlenecks
- Scale challenges

**Common deep dive topics:**
- Database sharding strategy
- Caching layer implementation
- Feed generation (push vs pull)
- Rate limiting
- Handling hot users (celebrities)

### **Example: Timeline Generation Deep Dive**

**Problem:** How to generate personalized timeline for user with 1000 followers?

**Option 1: Pull Model (Fanout on Read)**
- When user requests timeline:
  - Query all tweets from people they follow
  - Sort by timestamp
  - Return top N

**Pros:** Simple writes (just store tweet)
**Cons:** Slow reads (query multiple users each time)
**Use case:** When user follows many people

**Option 2: Push Model (Fanout on Write)**
- When user posts tweet:
  - Write tweet to all followers' pre-computed timelines
  - Each user has their own timeline inbox

**Pros:** Fast reads (timeline pre-computed)
**Cons:** Slow writes (write to 1M inboxes if user has 1M followers)
**Use case:** When user has few followers

**Hybrid Approach (Twitter\'s actual solution):**
- **Regular users:** Push model (fanout on write)
- **Celebrities:** Pull model (query at read time)
- Threshold: 1M followers

**Timeline request:**
1. Fetch pre-computed timeline (push model)
2. Fetch tweets from celebrities user follows (pull model)
3. Merge and sort
4. Cache result

### **Database Sharding Deep Dive**

**Problem:** Single MySQL can't handle 6K writes/sec.

**Sharding strategy:**
- Shard by **user_id** (consistent hashing)
- Tweets stored on same shard as author
- 100 shards → 60 writes/sec per shard ✅

**Querying timeline:**
- User's tweets: Single shard (fast)
- Timeline: Query multiple shards (followers on different shards)
  - Use scatter-gather or cache

**Hot spot handling:**
- Celebrity tweets hit one shard hard
- Solution: Replicate celebrity data across shards

### **Caching Deep Dive**

**What to cache:**
- User timelines (most accessed)
- Hot tweets (trending)
- User profiles

**Cache strategy:**
- **Cache-aside:** App checks cache first, then DB
- **TTL:** 5 minutes (balance freshness and load)
- **Eviction:** LRU (Least Recently Used)

**Cache sizing:**
- 100M active users
- 1KB per timeline
- 100 GB cache needed
- Redis cluster: 10 nodes × 10GB each

### **Time Check:** 40-50 minutes total

---

## Step 4: Wrap Up & Optimizations (5 min)

### **A. Discuss Failure Scenarios**

"What if...?"
- Server crashes → Load balancer routes to healthy servers
- Database fails → Standby takes over (automatic failover)
- Cache goes down → Requests go to database (slower but works)
- Entire datacenter fails → Multi-region deployment handles it

### **B. Monitoring & Operations**

- Metrics: QPS, latency (p50, p99), error rates
- Logging: Centralized logs (ELK stack)
- Alerting: PagerDuty for critical issues
- Dashboards: Grafana for visualization

### **C. Bottlenecks & Optimizations**

**Current bottlenecks:**
- Database writes (6K/sec approaching limits)
- Hot celebrity tweets (single shard hotspot)

**Optimizations:**
- Add more database shards
- Implement read replicas
- More aggressive caching
- CDN for media (offload traffic)

### **D. Trade-offs Made**

Summarize key decisions:
- Chose eventual consistency over strong (better availability)
- Chose Cassandra over MySQL (better write throughput)
- Hybrid push/pull for timeline (balance read/write performance)

### **E. Future Enhancements**

If we had more time:
- Machine learning recommendations
- Real-time trending topics
- Advanced analytics
- Video support

---

## Complete Example: Design URL Shortener

### **Step 1: Requirements (5 min)**

**Functional:**
- Shorten long URL → short URL
- Redirect short URL → original URL
- (Optional: Custom aliases, analytics)

**Non-functional:**
- 100M URLs shortened per month
- 1B redirects per month (10:1 read/write)
- Low latency (<50ms)
- High availability (99.9%)

**Assumptions:**
- Average URL: 100 characters
- Short URL: 7 characters
- Store forever (no expiration)

### **Step 2: High-Level Design (10 min)**

**Estimations:**
- Storage: 100M × 100 bytes = 10 GB/month = 600 GB (5 years)
- Write QPS: 100M/month ÷ 2.5M sec = 40 writes/sec
- Read QPS: 1B/month ÷ 2.5M = 400 reads/sec

**Components:**
- Load balancer
- API servers
- Database (PostgreSQL)
- Cache (Redis)

**APIs:**
\`\`\`
    POST / shorten
    Body: { long_url }
    Response: { short_url }

    GET /: short_code
  Redirect to long_url
        \`\`\`

**Database schema:**
\`\`\`
    url_mappings:
    - id (auto - increment)
        - short_code (unique index)
        - long_url
        - created_at
            \`\`\`

**Flow:**
1. POST /shorten → Generate short code → Store in DB → Return
2. GET /:code → Check cache → If miss, query DB → Redirect

### **Step 3: Deep Dive (20 min)**

**Short code generation:**

**Option 1: Hash (MD5/SHA)**
- MD5(long_url) → Take first 7 chars
- **Problem:** Collisions possible
- **Solution:** Append counter and rehash

**Option 2: Base62 encoding**
- Auto-increment ID (1, 2, 3...)
- Convert to base62 (0-9, a-z, A-Z)
- ID 125 → base62 "2b"
- **Pros:** No collisions, short codes
- **Cons:** Predictable (security concern)

**Chosen approach: Base62 with random starting point**

**Scaling:**
- 40 writes/sec → Single PostgreSQL handles easily
- 400 reads/sec → Cache hit rate 90% → Only 40 DB reads/sec ✅

**If scale 100×:**
- 4K writes/sec → Need sharding
- Shard by short_code hash
- 10 shards → 400 writes/sec each

### **Step 4: Wrap Up (5 min)**

**Monitoring:**
- Track redirect latency
- Cache hit rate
- Error rates

**Optimizations:**
- Add CDN for static assets
- Database replication for reads
- Rate limiting per user (prevent abuse)

**Time:** 40 minutes total ✅

---

## Key Principles

1. **Always follow the 4-step structure**
2. **Spend time proportionally**: Deep dive is most important
3. **Think out loud**: Explain your reasoning
4. **Draw diagrams**: Visual communication
5. **Calculate numbers**: Validate your decisions
6. **Discuss trade-offs**: Show you understand pros/cons
7. **Adapt to feedback**: Interviewer hints are valuable
8. **Manage time**: Check in at each step

**Practice this framework until it becomes second nature!**`,
};
