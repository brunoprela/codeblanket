/**
 * Quiz questions for PACELC Theorem section
 */

export const pacelctheoremQuiz = [
  {
    id: 'pacelc-disc-q1',
    question:
      "You're designing a ride-sharing app like Uber. Driver locations must be updated in real-time (every 2-3 seconds) and displayed to nearby riders. Should you use a PA/EL system (Cassandra) or PC/EC system (HBase)? Discuss the trade-offs and explain your choice using PACELC framework.",
    sampleAnswer: `I would choose a **PA/EL system (Cassandra or DynamoDB)** for real-time driver location tracking. The EL choice (Latency over Consistency during normal operation) is critical here because the system requires sub-second latency for a good user experience, and perfect consistency is not required.

**Requirements Analysis:**

**Latency requirement**: 
- Need to display driver locations updated every 2-3 seconds
- Riders expect real-time map updates
- Any latency >500ms degrades UX

**Consistency requirement**:
- Driver location being 1-2 seconds stale is acceptable
- Better to show slightly outdated location than no location
- Eventual consistency is fine

**PACELC Choice: PA/EL (Cassandra/DynamoDB)**

**During Partition (PA):**
\`\`\`
Scenario: Network partition between US East and West datacenters

PA behavior:
- Both datacenters continue accepting driver location updates
- Riders in each region see locations from their datacenter
- Data temporarily inconsistent across regions
- After partition heals, data converges

Why this is acceptable:
- Better for riders to see slightly stale driver locations than no drivers
- Driver in SF partition unavailable to NYC rider anyway (too far)
- Availability critical for ride-sharing (people need rides NOW)
\`\`\`

**Normal Operation (EL):**
\`\`\`
EL behavior:
- Read driver locations from nearest replica
- Sub-millisecond latency (1-5ms)
- Location might be 1-2 seconds stale if replica hasn't caught up
- Writes are async, don't wait for all replicas

Why this is acceptable:
- Drivers moving ~15 m/s (city speed)
- 2 seconds stale = 30 meters off (acceptable on map)
- Low latency critical for smooth map experience
- Users don't notice 1-2 second staleness
\`\`\`

**Why NOT PC/EC (HBase):**

**PC/EC problems for this use case:**
\`\`\`
Consistency cost:
- Must read from authoritative source (higher latency: 10-50ms)
- Must wait for write replication (10-100ms write latency)
- At 1M driver updates/sec, this becomes bottleneck

Availability cost:
- During partition, some regions become unavailable
- Riders can't see ANY drivers (unacceptable)

Result: Slower, less available, no real benefit
(Strong consistency doesn't matter for driver locations)
\`\`\`

**Architecture Design:**

\`\`\`
Cassandra (PA/EL):
- Partition key: (geohash, driver_id)
- Replication factor: 3
- Consistency level: ONE (for reads and writes)

Write path:
- Driver app sends location every 2 seconds
- Write to nearest Cassandra node
- Async replication to other replicas
- Write latency: 5-10ms

Read path:
- Rider app requests drivers near location
- Query local Cassandra replica
- Get results in 1-5ms
- Might be 1-2 seconds stale (acceptable)

Result:
- Sub-5ms read latency (smooth map)
- Handles 1M+ updates/sec
- Always available
- Eventual consistency acceptable
\`\`\`

**Trade-offs Accepted:**

**✅ Benefits (PA/EL):**
- Sub-5ms latency for map updates
- Always available (riders always see drivers)
- Scales to millions of drivers
- Handles partitions gracefully

**❌ Trade-offs (PA/EL):**
- Driver location might be 1-2 seconds stale
- During partition, different regions see different data temporarily
- Rare edge case: Rider sees driver location, but driver already moved

**Why Trade-offs Are Acceptable:**

1. **Staleness is fine**: Driver locations naturally become stale (drivers are moving). 2 seconds is negligible.

2. **Availability is critical**: Riders need to see drivers to request rides. No drivers visible = no business.

3. **Latency matters**: Smooth map updates require <5ms. 50ms would feel laggy.

**Real-World Validation:**

Uber actually uses **Cassandra and Redis** for driver location tracking:
- Cassandra: Persistent storage (PA/EL)
- Redis: In-memory cache (PA/EL)
- Both prioritize availability and low latency

They DON'T use HBase (PC/EC) because:
- Strong consistency not needed
- Higher latency unacceptable
- Less availability unacceptable

**Key Insight:**

For real-time location tracking, the **normal operation trade-off (ELC)** is more important than partition behavior. The system operates normally 99.9% of the time, and that's when users experience the latency. Choosing EL (low latency, eventual consistency) provides better UX than EC (high latency, strong consistency) for this use case.`,
    keyPoints: [
      'Real-time location tracking requires <5ms latency, making PA/EL (Cassandra) ideal',
      'EL choice (Latency over Consistency) critical for smooth user experience',
      'Driver location staleness (1-2 seconds) is acceptable and natural',
      'PA choice (Availability during partition) ensures riders always see drivers',
      'PC/EC (HBase) would have 10-50ms latency and lower availability (unacceptable)',
      'Uber uses Cassandra/Redis (PA/EL systems) in production for this exact reason',
      'Normal operation behavior (ELC) more important than partition behavior for UX',
    ],
  },
  {
    id: 'pacelc-disc-q2',
    question:
      'Explain to a product manager why Google Spanner (PC/EC) has 50-100ms commit latency while DynamoDB (PA/EL) has <10ms latency. The PM argues "we should use DynamoDB for everything since it\'s faster." How would you respond using PACELC framework?',
    sampleAnswer: `Great question! The latency difference comes from **different PACELC trade-offs** for different use cases. Let me explain why both databases exist and when to use each.

**Why Spanner is Slower (PC/EC):**

Google Spanner chooses **PC/EC** - prioritizing strong consistency both during partitions and normal operation.

**Normal Operation (EC - Consistency over Latency):**
\`\`\`
Spanner commit process:
1. Write proposal sent to Paxos group
2. Wait for majority acknowledgment (cross-datacenter)
3. Wait for TrueTime uncertainty window (atomic clocks)
4. Commit confirmed

Time breakdown:
- Cross-datacenter Paxos: 20-50ms (network)
- TrueTime uncertainty: 1-7ms (atomic clock sync)
- Write to disk: 5-10ms
- Total: 50-100ms

Why Spanner does this:
- Guarantees external consistency (stricter than strong consistency)
- All commits globally ordered
- Strong consistency across all datacenters globally
\`\`\`

**Why DynamoDB is Faster (PA/EL):**

DynamoDB chooses **PA/EL** - prioritizing low latency and availability.

**Normal Operation (EL - Latency over Consistency):**
\`\`\`
DynamoDB write process:
1. Write to local node
2. Acknowledge immediately
3. Async replication to other replicas

Time breakdown:
- Local write: 1-5ms
- Acknowledge: <10ms
- (Replication happens in background)

Why DynamoDB does this:
- Optimizes for low latency
- Eventually consistent (data converges in milliseconds)
- Good enough for most use cases
\`\`\`

**So Why Not Use DynamoDB for Everything?**

**Use Case 1: Google Ads Billing (Spanner - PC/EC)**

\`\`\`
Requirement: Charge advertisers accurately

Problem with DynamoDB (PA/EL):
- Eventual consistency means ad impressions might be double-counted
- During partition, different regions might bill differently
- Financial data must be strongly consistent

Example scenario:
- Advertiser has $100 budget
- US datacenter shows 1000 impressions ($50 spent)
- EU datacenter shows 1100 impressions ($55 spent) (stale)
- Eventual consistency causes billing discrepancy

With Spanner (PC/EC):
- All datacenters agree on impression count
- Budget tracking accurate globally
- 50-100ms latency acceptable (not user-facing)
- Correctness more important than speed
\`\`\`

**Use Case 2: E-commerce Product Catalog (DynamoDB - PA/EL)**

\`\`\`
Requirement: Display product information fast

Benefits of DynamoDB (PA/EL):
- Product details rarely change
- Showing slightly stale price (1-2 seconds) acceptable
- <10ms latency for fast page loads
- High availability critical (product page must always load)

Why Spanner would be WORSE:
- 50-100ms page load feels sluggish
- Users bounce if site is slow
- Strong consistency not needed (price doesn't change every second)
- No business value from waiting 50ms for same data
\`\`\`

**Decision Framework:**

**Use Spanner (PC/EC) when:**

1. **Strong consistency required**: Financial transactions, billing, inventory
2. **Global coordination needed**: Distributed locks, leader election
3. **Correctness > Speed**: Better to be slow and right than fast and wrong
4. **Not user-facing**: Background jobs, batch processing

**Examples:**
- Google Ads billing
- Bank account transfers
- Inventory management
- Financial trading

**Use DynamoDB (PA/EL) when:**

1. **Low latency critical**: User-facing reads, API responses
2. **High availability required**: Must always work
3. **Eventual consistency acceptable**: Slight staleness okay
4. **High throughput needed**: Millions of requests/sec

**Examples:**
- Product catalogs
- User profiles
- Shopping carts
- Social media feeds
- Session storage

**Cost Comparison:**

**Spanner:**
- Latency: 50-100ms (commit)
- Cost: ~$1,000-5,000/month (minimum 3 nodes)
- Throughput: Thousands of QPS per node
- **When worth it**: Financial data, global consistency required

**DynamoDB:**
- Latency: <10ms (local)
- Cost: $0.25 per million reads (on-demand)
- Throughput: Unlimited (auto-scaling)
- **When worth it**: User-facing data, high scale

**Real-World Hybrid Architecture:**

\`\`\`
Google's Approach:

Spanner (PC/EC):
- Ads billing (must be accurate)
- Financial transactions
- User authentication (security critical)

BigTable (PA/EL):
- Gmail storage (eventual consistency fine)
- Search index (slightly stale acceptable)
- YouTube metadata (availability important)

Result: Right tool for each job
\`\`\`

**Key Message to PM:**

"We shouldn't use DynamoDB for everything because **latency is not the only requirement**. For financial data (billing, payments, inventory), we need **strong consistency** - better to be 50ms slower and correct than 10ms fast and wrong. For user-facing data (product pages, profiles), we need **low latency** - DynamoDB's 10ms is perfect.

The 50-100ms Spanner latency buys us **global strong consistency** which is critical for financial accuracy. The 10ms DynamoDB latency provides **user experience** for non-critical data. We should use both: Spanner for financial/critical data, DynamoDB for user-facing/non-critical data.

**Analogy**: It's like asking 'why not use sports cars for everything since they're faster?' Because sometimes you need a truck (to carry heavy loads) and sometimes you need a sports car (to go fast). Same with databases - use the right tool for the job."`,
    keyPoints: [
      'Spanner (PC/EC) chooses Consistency over Latency, resulting in 50-100ms for strong global consistency',
      'DynamoDB (PA/EL) chooses Latency over Consistency, resulting in <10ms with eventual consistency',
      'Use Spanner (PC/EC) for financial data requiring strong consistency (billing, payments)',
      'Use DynamoDB (PA/EL) for user-facing data where latency matters (catalogs, profiles)',
      'The latency difference reflects different trade-offs, not better/worse technology',
      'Real-world systems use both: Spanner for critical data, DynamoDB for high-volume data',
      'Choose based on requirements: correctness vs speed, consistency vs latency',
    ],
  },
  {
    id: 'pacelc-disc-q3',
    question:
      'How would you use PACELC theorem to decide between Cassandra and HBase for a social media analytics platform that tracks post engagement metrics (likes, shares, views) in real-time? Consider both partition and normal operation scenarios.',
    sampleAnswer: `I would choose **Cassandra (PA/EL)** for a social media analytics platform. The key insight is that the **normal operation trade-off (ELC)** dominates this decision because the system operates normally 99.9% of the time.

**Requirements Analysis:**

**Functional Requirements:**
- Track engagement metrics (likes, shares, views) in real-time
- Update metrics every few seconds as users engage
- Display metrics on posts (1M+ posts viewed per second)
- Aggregate metrics for dashboards and reports

**Non-Functional Requirements:**
- **High write throughput**: Millions of engagement events per second
- **Low read latency**: Metrics displayed on every post view (<50ms)
- **High availability**: Analytics dashboard must always work
- **Eventual consistency acceptable**: Showing 100 likes vs 103 likes (actual) is fine

**PACELC Analysis:**

**Scenario 1: Normal Operation (99.9% of time)**

This is where **ELC trade-off** matters most.

**Cassandra (PA/EL - Latency prioritized):**
\`\`\`
Write path:
- User likes post
- Write to nearest Cassandra node (1-5ms)
- Acknowledge immediately
- Async replication to other replicas
- Write latency: <10ms
- Throughput: 10K+ writes/sec per node

Read path:
- User views post
- Query nearest replica for like count
- Read latency: 1-5ms
- Might see 100 likes (actual: 103, 3 likes replicating)
- Eventual consistency: Converges in <1 second

Result: Fast, scales easily
\`\`\`

**HBase (PC/EC - Consistency prioritized):**
\`\`\`
Write path:
- User likes post
- Write must go to RegionServer
- Wait for WAL sync + replication
- Write latency: 10-50ms
- Throughput: 1K-5K writes/sec per node

Read path:
- User views post
- Query must go to RegionServer (authoritative)
- May not be nearest node
- Read latency: 10-50ms
- Always shows correct count (103 likes)
- Strong consistency

Result: Slower, harder to scale
\`\`\`

**For normal operation:**
- Cassandra: <10ms reads, <10ms writes, scales horizontally
- HBase: 10-50ms reads, 10-50ms writes, vertical scaling complex

**Winner for normal operation: Cassandra (EL - prioritizes latency)**

**Why latency matters:**
- Users view millions of posts per second
- Each view requires reading metrics
- 10-50ms latency (HBase) would slow page loads
- 1-5ms latency (Cassandra) provides smooth experience

**Why eventual consistency is acceptable:**
- Like count being off by 2-3 (1 second replication lag) doesn't matter
- Users don't notice 100 vs 103 likes
- Metrics converge quickly

---

**Scenario 2: Network Partition (0.1% of time)**

**Cassandra (PA/EL - Availability prioritized):**
\`\`\`
Scenario: Partition between US and EU datacenters

PA behavior:
- Both datacenters continue accepting writes
- US users' likes recorded in US datacenter
- EU users' likes recorded in EU datacenter
- Like counts temporarily diverge
- After partition heals, counts merge (eventual consistency)

Example:
- Post has 100 likes pre-partition
- During partition: US datacenter records 10 likes, EU records 15 likes
- US shows 110, EU shows 115
- After heal: Both show 125 likes

Impact:
- ✅ Analytics dashboard continues working
- ✅ Users can like/share posts
- ❌ Like counts temporarily inconsistent (acceptable for analytics)
\`\`\`

**HBase (PC/EC - Consistency prioritized):**
\`\`\`
Scenario: Partition between US and EU datacenters

PC behavior:
- Only datacenter with master continues working
- Minority partition becomes read-only or unavailable
- Ensures like counts remain consistent

Example:
- Master in US datacenter
- EU datacenter loses connection
- EU users cannot like/share posts (unavailable)
- EU analytics dashboard shows stale data or errors

Impact:
- ❌ EU users cannot engage with posts
- ❌ EU analytics unavailable
- ✅ Like counts remain consistent
\`\`\`

**For partition:**
- Cassandra: Continues working, counts temporarily inconsistent
- HBase: Minority partition unavailable

**Winner for partition: Cassandra (PA - prioritizes availability)**

**Why availability matters during partition:**
- Users expect social media to always work
- Better to show slightly inconsistent like counts than no dashboard
- Engagement (likes/shares) must continue during network issues

---

**Data Model Design:**

**Cassandra Schema (Optimized for PA/EL):**
\`\`\`
Table: post_engagement
Partition key: post_id
Clustering key: metric_type (likes, shares, views)

CREATE TABLE post_engagement (
  post_id UUID,
  metric_type TEXT,
  count COUNTER,
  PRIMARY KEY (post_id, metric_type)
);

Write:
UPDATE post_engagement 
SET count = count + 1 
WHERE post_id = X AND metric_type = 'likes';
(CL=ONE, async replication, fast)

Read:
SELECT count FROM post_engagement 
WHERE post_id = X AND metric_type = 'likes';
(CL=ONE, nearest replica, fast)
\`\`\`

**Why This Works:**
- Counter columns handle concurrent increments (conflicts resolve automatically)
- CL=ONE provides low latency
- Eventual consistency acceptable for analytics
- Scales horizontally (add nodes for more throughput)

---

**HBase Alternative (PC/EC):**
\`\`\`
Table: post_engagement
Row key: post_id
Column family: metrics
Columns: likes, shares, views

Put:
put 'post_engagement', post_id, 'metrics:likes', count
(Wait for WAL + replication, slower)

Get:
get 'post_engagement', post_id, 'metrics:likes'
(Query RegionServer, authoritative, slower)
\`\`\`

**Why This is Worse:**
- Higher latency for both reads and writes
- Sharding complex (hot posts cause hotspots)
- Strong consistency not needed for analytics

---

**Trade-off Summary:**

**Cassandra (PA/EL):**
- ✅ <10ms read/write latency (smooth UX)
- ✅ Always available (works during partitions)
- ✅ Horizontal scaling (add nodes for more throughput)
- ✅ Handles millions of writes/sec
- ❌ Like counts might be off by 2-3 temporarily (acceptable)
- ❌ During partition, different regions see different counts briefly

**HBase (PC/EC):**
- ✅ Like counts always accurate (strong consistency)
- ✅ No conflicting counts during partitions
- ❌ 10-50ms latency (slower UX)
- ❌ Unavailable during partitions (minority partition)
- ❌ Harder to scale horizontally
- ❌ Lower throughput (1K-5K writes/sec per node)

**For social media analytics: Cassandra's trade-offs are clearly better**

---

**Real-World Validation:**

**Instagram uses Cassandra** for engagement metrics:
- Handles billions of likes/comments per day
- Low latency for smooth app experience
- High availability critical
- Eventual consistency acceptable

**Instagram does NOT use HBase because:**
- Strong consistency not needed (like count off by 3 is fine)
- Higher latency would degrade UX
- Lower availability unacceptable

---

**Key Insight:**

The **normal operation trade-off (ELC)** is decisive here. Since the system operates normally 99.9% of the time, optimizing for that state is critical. Cassandra's choice of **Latency over Consistency (EL)** provides better UX and scale. HBase's choice of **Consistency over Latency (EC)** provides no real benefit because perfect accuracy of like counts is not a business requirement.

The partition behavior **(PA vs PC)** reinforces this: analytics dashboards must remain available during rare network issues, making PA (availability) the right choice over PC (consistency).`,
    keyPoints: [
      'Cassandra (PA/EL) ideal for analytics: low latency, high availability, handles massive write throughput',
      'EL choice (Latency over Consistency) provides <10ms reads/writes vs 10-50ms for HBase',
      'Normal operation behavior (99.9% of time) more important than partition behavior (0.1%)',
      "Eventual consistency acceptable: like count off by 2-3 doesn't impact UX",
      'PA choice (Availability during partition) keeps analytics dashboard working',
      'HBase (PC/EC) provides strong consistency but no business value for like counts',
      'Instagram uses Cassandra for this exact use case in production',
    ],
  },
];
