/**
 * Latency vs Throughput Section
 */

export const latencyvsthroughputSection = {
  id: 'latency-vs-throughput',
  title: 'Latency vs Throughput',
  content: `Understanding the difference between **latency** and **throughput** and knowing when to optimize for each is critical in system design. These metrics represent fundamentally different aspects of performance.

## Definitions

**Latency**: The time it takes to complete a **single operation**.
- Measured in: milliseconds (ms) or seconds
- Examples: "API response time is 50ms", "Database query takes 200ms"
- User-facing metric: "How fast does this feel?"

**Throughput**: The number of operations completed in a **given time period**.
- Measured in: requests per second (RPS), queries per second (QPS), transactions per second (TPS)
- Examples: "System handles 10,000 requests/second", "Database processes 5,000 transactions/second"
- System-facing metric: "How much work can we do?"

---

## The Core Trade-off

**You can't always optimize for both.** Sometimes improving one hurts the other.

### Example: Restaurant Analogy

**Low Latency Restaurant** (Fast Food):
- Goal: Serve each customer quickly
- Latency: 3 minutes per customer ⚡
- Throughput: 20 customers/hour
- Trade-off: Simple menu, limited customization

**High Throughput Restaurant** (Buffet):
- Goal: Serve many customers per hour
- Latency: 30 minutes per customer (wait in line, wait for food)
- Throughput: 200 customers/hour 📈
- Trade-off: Customers wait longer, but more total customers served

---

## When to Optimize for Latency

### User-Facing Interactive Systems

**Prioritize latency when:**
- Humans are waiting for response
- Real-time or near-real-time requirements
- User experience depends on speed
- Each request is independent

### Use Cases

**1. API Endpoints for Web/Mobile Apps**

**Example**: User clicks "Search" button on e-commerce site

**Requirements**:
- Response time < 200ms (perceived as instant)
- Each user's request independent
- Bad latency = users leave site

**Optimization**:
- Cache frequently accessed data (Redis)
- CDN for static assets
- Database indexes for fast queries
- Minimize network round trips

**Metrics**:
- P50 latency: 50ms (50% of requests faster than this)
- P95 latency: 150ms (95% of requests faster than this)
- P99 latency: 300ms (99% of requests faster than this)

**2. Real-Time Trading Systems**

**Example**: Stock trading platform

**Requirements**:
- Every millisecond matters
- Low latency = competitive advantage
- Latency: <10ms (sub-millisecond for HFT)

**Optimization**:
- In-memory data structures
- Colocation near exchanges
- Dedicated low-latency network
- Lock-free algorithms

**3. Video Games**

**Example**: Online multiplayer game

**Requirements**:
- Player actions must feel instant
- Latency > 100ms = "laggy" experience
- Each player's action independent

**Optimization**:
- Regional game servers (reduce geographic latency)
- UDP instead of TCP (lower latency, sacrifice reliability)
- Client-side prediction
- Delta compression

---

## When to Optimize for Throughput

### Batch Processing and Background Jobs

**Prioritize throughput when:**
- Processing large datasets
- Background jobs (not user-facing)
- Total work completed matters more than speed per item
- Can batch operations together

### Use Cases

**1. Data Pipeline / ETL Jobs**

**Example**: Nightly job processing 10 million user events

**Requirements**:
- Must complete before morning (6-hour window)
- Processing 10M events = throughput priority
- No user waiting for individual event

**Optimization**:
- Batch processing (process 10,000 events at once)
- Parallel workers (100 workers processing simultaneously)
- Optimize for CPU/memory efficiency
- Sacrifice per-item latency for total throughput

**Metrics**:
- Throughput: 500,000 events/minute
- Individual event latency: Not measured (nobody waiting)

**2. Video Encoding**

**Example**: YouTube encoding uploaded videos

**Requirements**:
- Must encode millions of videos per day
- Each video takes minutes to encode
- No user waiting in real-time

**Optimization**:
- Distributed workers (thousands of encoding machines)
- Batch scheduling
- Optimize for parallel processing
- High throughput = cost efficiency

**3. Log Processing**

**Example**: Processing application logs for analytics

**Requirements**:
- Process billions of log entries per day
- Aggregate metrics (not real-time)
- Total processed matters, not individual log latency

**Optimization**:
- Stream processing (Kafka, Flink)
- Micro-batching (process 1,000 logs at a time)
- Compression for network efficiency
- Parallel consumers

---

## Real-World Examples

### Example 1: Database Queries

**Scenario A: User Login (Latency-Critical)**

\`\`\`sql
SELECT * FROM users WHERE email = 'user@example.com';
\`\`\`

**Optimization**:
- Index on email column (instant lookup)
- Read replica to avoid querying primary
- Cache user session (avoid repeated queries)
- **Goal**: <10ms query time

**Scenario B: Analytics Report (Throughput-Critical)**

\`\`\`sql
SELECT country, COUNT(*), AVG(age) 
FROM users 
GROUP BY country;
\`\`\`

**Optimization**:
- Run on read replica (not primary)
- Columnar storage for fast aggregations
- Parallel query execution
- **Goal**: Process 100M rows in 5 minutes

**Different queries, different priorities.**

---

### Example 2: Payment Processing

**Scenario A: Payment API (Latency-Critical)**

User clicks "Pay Now" → Authorize payment

**Requirements**:
- User waiting for response
- Must respond in <500ms
- One payment at a time

**Optimization**:
- Fast payment gateway API
- Async processing (authorize quickly, settle later)
- Minimal database writes
- **Metric**: P95 latency <500ms

**Scenario B: Payment Settlement (Throughput-Critical)**

Background job settling 10 million payments overnight

**Requirements**:
- Must settle all payments before morning
- No user waiting
- Batch processing acceptable

**Optimization**:
- Batch settlements (send 10,000 payments to bank at once)
- Parallel workers
- Optimize for throughput
- **Metric**: 50,000 settlements/minute

---

## Batching: Trading Latency for Throughput

**Batching** increases throughput at the cost of latency.

### Example: Sending Emails

**No Batching (Low Latency)**:
- Send each email immediately
- Latency: 100ms per email
- Throughput: 10 emails/second
- Overhead: Connection setup/teardown for each email

**Batching (High Throughput)**:
- Collect 1,000 emails, send together
- Latency: 5 seconds (wait to collect batch) + 10 seconds (send batch) = 15 seconds per email
- Throughput: 1,000 emails/15 seconds = 66 emails/second
- Benefit: 6.6x higher throughput!

**Trade-off**: Individual emails take longer (15s vs 100ms), but you can send more total emails.

**When to use batching**:
- Background jobs (email campaigns, notifications)
- Database inserts (bulk insert 1,000 rows vs 1,000 individual inserts)
- Log shipping (batch logs before sending to aggregation service)

---

## Little's Law

**Little's Law** connects latency, throughput, and concurrency:

\`\`\`
Throughput = Concurrency / Latency
\`\`\`

Or rearranged:

\`\`\`
Concurrency = Throughput × Latency
\`\`\`

### Example: API Server

**Given**:
- Latency: 100ms (0.1 seconds) per request
- Desired throughput: 1,000 requests/second

**Calculate required concurrency**:
\`\`\`
Concurrency = 1,000 req/s × 0.1s = 100 concurrent requests
\`\`\`

**Implication**: You need to handle 100 concurrent requests to achieve 1,000 req/s at 100ms latency.

**Scaling options**:
1. Reduce latency to 50ms → Need only 50 concurrent requests
2. Increase concurrency to 200 → Can handle 2,000 req/s
3. Add more servers → Distribute load

---

## Measuring Latency: Percentiles Matter

**Don't use average latency.** Use percentiles.

### Why Averages Are Misleading

**Scenario**: 100 API requests

- 95 requests: 50ms (fast)
- 5 requests: 1,000ms (slow, outliers)

**Average latency**: (95 × 50ms + 5 × 1,000ms) / 100 = **97.5ms**

**Sounds great!** But 5% of users waited 1 second.

### Use Percentiles

- **P50 (median)**: 50% of requests faster than this (50ms)
- **P95**: 95% of requests faster than this (60ms)
- **P99**: 99% of requests faster than this (200ms)
- **P99.9**: 99.9% of requests faster than this (1,000ms)

**Why P99 matters**: If user makes 100 API calls, statistically 1 call hits P99 latency. That slow call can block the entire user experience.

**Example - Amazon**: If user loads product page that makes 100 backend API calls, and P99 latency is 1 second, the page load will likely hit 1+ slow calls, making the entire page slow.

---

## Architectural Patterns

### Optimize for Latency

**1. Caching**
- Redis, Memcached for frequently accessed data
- Cache-aside pattern (check cache first, DB on miss)

**2. Indexing**
- Database indexes (B-tree, hash) for fast lookups
- Reduces query latency from 500ms to 5ms

**3. CDN (Content Delivery Network)**
- Serve static assets from edge locations near users
- Reduces geographic latency

**4. Read Replicas**
- Distribute reads across multiple replicas
- Reduces load on primary, faster reads

**5. In-Memory Processing**
- Keep hot data in memory (Redis, in-process cache)
- Avoid disk I/O (disk: ~5ms, memory: ~0.001ms)

### Optimize for Throughput

**1. Batching**
- Batch database inserts, API calls, message sends
- Reduces per-item overhead

**2. Parallel Processing**
- Multiple workers processing simultaneously
- Horizontal scaling (more machines)

**3. Asynchronous Processing**
- Queue-based architecture (don't wait for response)
- Message queues (Kafka, RabbitMQ, SQS)

**4. Connection Pooling**
- Reuse database connections (avoid connection overhead)
- Increases throughput significantly

**5. Compression**
- Compress data before sending over network
- Trades CPU (compression) for network throughput

---

## Common Mistakes

### ❌ Mistake 1: Optimizing Throughput for Latency-Critical Systems

**Example**: User clicks "Login" → System batches login requests

**Problem**: User waits 5 seconds while batch fills up. Terrible UX.

**Fix**: For user-facing requests, optimize for latency.

### ❌ Mistake 2: Optimizing Latency for Throughput-Critical Systems

**Example**: Processing 10M log entries one-by-one with minimal latency

**Problem**: Takes 10x longer than batching. Wastes resources.

**Fix**: For batch jobs, optimize for throughput.

### ❌ Mistake 3: Using Average Instead of Percentiles

**Problem**: Average hides outliers. 1% of users get terrible experience.

**Fix**: Always measure P95, P99, P99.9 latency.

### ❌ Mistake 4: Ignoring Little's Law

**Problem**: Want 10,000 req/s but only have 10 concurrent threads and 100ms latency.

**Math**: 10 threads / 0.1s = 100 req/s (100x too slow!)

**Fix**: Increase concurrency to 1,000 threads or reduce latency to 1ms.

---

## Best Practices

### ✅ 1. Identify Critical Path

Determine if system is:
- **Latency-critical**: User waiting (optimize for speed per request)
- **Throughput-critical**: Background job (optimize for total work done)

### ✅ 2. Measure Both

Always measure both metrics:
- Latency: P50, P95, P99
- Throughput: Requests per second

### ✅ 3. Set SLOs (Service Level Objectives)

Examples:
- "P99 latency < 500ms for all API endpoints"
- "Process 1M events/minute during peak hours"

### ✅ 4. Use Different Strategies for Different Data Paths

**Example - Twitter**:
- **Tweet loading** (latency): < 100ms, cached, indexed
- **Tweet indexing** (throughput): Batch processing, millions/second

### ✅ 5. Monitor in Production

Use APM tools (Datadog, New Relic) to track:
- Latency percentiles over time
- Throughput over time
- Identify regressions

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd prioritize **[latency/throughput]** because:

**Use case analysis**: [Describe if user-facing or batch]

**Latency approach** (if latency-critical):
- Cache frequently accessed data
- Database indexes
- CDN for static assets
- Target: P99 < 200ms

**Throughput approach** (if throughput-critical):
- Batch processing
- Parallel workers
- Asynchronous processing
- Target: 100,000 operations/minute

**Trade-off**: [Explain what we sacrifice]"

### Example: Design Video Upload System

**Answer**: 

"Video upload has two paths with different priorities:

**Upload API (latency-critical)**:
- User uploads video, waits for confirmation
- Optimize for fast upload: presigned S3 URLs, chunked upload
- Target: <5 seconds for upload confirmation

**Video Encoding (throughput-critical)**:
- Background job, user doesn't wait
- Optimize for throughput: batch encoding, 1,000 parallel workers
- Target: Encode 10,000 videos/hour

**Trade-off**: Video encoding has high latency per video (5 minutes), but high throughput (10,000/hour). User doesn't care about per-video latency since they're not waiting."

---

## Summary Table

| Aspect | Latency | Throughput |
|--------|---------|-----------|
| **Definition** | Time per operation | Operations per time period |
| **Measured in** | Milliseconds, seconds | Req/sec, TPS, QPS |
| **Optimize for** | User-facing, real-time | Batch jobs, background |
| **Techniques** | Caching, indexing, CDN | Batching, parallelism |
| **Example** | API response time | ETL processing rate |
| **User Impact** | Feels fast | System can handle load |
| **Metrics** | P50, P95, P99 | RPS, TPS, QPS |
| **Trade-off** | May reduce throughput | May increase latency |

---

## Key Takeaways

✅ Latency = time per operation, Throughput = operations per time
✅ User-facing systems: Optimize for latency (P99 < 200ms)
✅ Background jobs: Optimize for throughput (maximize work done)
✅ Batching trades latency for throughput (good for background tasks)
✅ Little's Law: Throughput = Concurrency / Latency
✅ Use percentiles (P95, P99), not averages, to measure latency
✅ Different parts of same system may optimize differently
✅ In interviews, identify if system is latency or throughput-critical first`,
};
