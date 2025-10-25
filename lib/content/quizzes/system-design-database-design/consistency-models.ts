/**
 * Quiz questions for Consistency Models section
 */

export const consistencymodelsQuiz = [
  {
    id: 'consistency-models-disc-q1',
    question:
      "You are designing a collaborative document editing system like Google Docs. Multiple users can edit the same document simultaneously. What consistency model should you use and why? Discuss how you'd handle conflicting edits.",
    sampleAnswer: `For a collaborative editing system like Google Docs, I would use **Causal Consistency with Operational Transformation (OT) or CRDTs** for conflict resolution.

**Why Causal Consistency:**

**Requirements Analysis:**
- Multiple users editing simultaneously (high concurrency)
- Edits must appear in sensible order (causality matters)
- Low latency critical (typing must feel instant)
- High availability (users must work offline)
- Linearizability too restrictive (would require locking entire document)

**Causal Consistency Benefits:**

1. **Preserves Intent**: If User A types "hello" then User B replies "world" (seeing "hello"), causal consistency ensures all users see "hello" before "world"

2. **Allows Concurrent Edits**: Users A and B can simultaneously edit different parts of the document without waiting for each other

3. **Low Latency**: No coordination required for concurrent (non-causally-related) edits

**Architecture:**

\`\`\`
Client-Side:
- Local edits apply immediately (optimistic update)
- Track edit history with version vectors
- Detect causal relationships

Server-Side:
- Accept edits from all users
- Track causality using version vectors or vector clocks
- Broadcast edits to other users
- Order by causal dependencies

Conflict Resolution:
- Use Operational Transformation (OT) or CRDTs
- Transform concurrent edits to preserve intent
- Converge to consistent state
\`\`\`

**Example Scenario:**

\`\`\`
Document: "The cat"

Time T1:
- User A at position 4: inserts "quick " → "The quick cat"
- User B at position 8: inserts " sat" → "The cat sat"

Both edits concurrent (neither saw the other's edit)

Causal Consistency allows both
Eventually must converge to: "The quick cat sat"

Operational Transformation:
- Transform B's edit: Original position 8, but A inserted at position 4
- Adjust B's position: 8 + 6 (length of A's insert) = 14
- Result: "The quick cat sat" ✓
\`\`\`

**Why NOT Linearizability:**

\`\`\`
Problem with Strong Consistency:
- Would require locking entire document for each keystroke
- User A types → locks document → applies edit → unlocks
- User B types → waits for lock → applies edit
- Result: Terrible UX, feels laggy

Also breaks offline editing (partition tolerance)
\`\`\`

**Why NOT Eventual Consistency (alone):**

\`\`\`
Problem:
- No ordering guarantees
- Could show User B's reply before User A's original text
- Confusing UX
- Still need conflict resolution, but no causality tracking
\`\`\`

**Conflict Resolution Approaches:**

**Option 1: Operational Transformation (OT)**
- Used by Google Docs
- Transform operations to account for concurrent edits
- Requires complex transformation functions
- Provides strong eventual consistency

\`\`\`
Example:
Operation A: Insert "X" at position 5
Operation B: Delete character at position 3

Transform B when applied after A:
- A inserted at 5, so positions after 5 shift
- B's position 3 unaffected (before A)
- B executes as-is: Delete at position 3
\`\`\`

**Option 2: CRDTs (Conflict-Free Replicated Data Types)**
- Used by newer systems (Figma, Notion)
- Data structures that automatically resolve conflicts
- No transformation logic needed
- Types: LWW-Element-Set, RGA (Replicated Growable Array)

\`\`\`
Example: RGA for text
- Each character has unique ID + position
- Concurrent inserts use character IDs for ordering
- Deletes mark characters as tombstones
- All replicas converge to same state automatically
\`\`\`

**Real-World Implementation (Google Docs):**

\`\`\`
Architecture:
- Client-side: Immediate local updates (optimistic)
- Operation queue: Track all edits with metadata
- Server: Central coordination, broadcasts operations
- Version vectors: Track causality
- OT: Transform concurrent operations
- Periodic snapshots: Avoid replaying entire history

Consistency Model:
- Causal consistency for edit ordering
- Eventual convergence through OT
- Session consistency (user sees own edits immediately)
\`\`\`

**Trade-Offs Accepted:**

✅ **Benefits:**
- Instant feedback (low latency)
- Offline editing support
- Scales to many concurrent users
- No edit blocking/locking

❌ **Trade-Offs:**
- Complex conflict resolution logic
- Occasional "surprising" conflict resolutions
- Must handle intentional conflicts (two users editing same word)
- Cannot prevent conflicts, only resolve them

**Key Insight:**

Collaborative editing is a perfect use case for **Causal Consistency** because:
1. **Causality matters**: User reactions should appear after original content
2. **Concurrency needed**: Multiple users editing simultaneously
3. **Linearizability too strict**: Would serialize all edits (terrible UX)
4. **Eventual too weak**: Needs causal ordering for sensible results

The combination of Causal Consistency + OT/CRDTs provides the right balance: preserve meaningful order (causality) while allowing maximum concurrency.`,
    keyPoints: [
      'Causal Consistency ideal for collaborative editing - preserves intent while allowing concurrency',
      'Operational Transformation (OT) or CRDTs resolve conflicting concurrent edits',
      'Linearizability too restrictive - would require locking, terrible UX',
      'Eventual Consistency alone insufficient - needs causal ordering',
      'Client-side optimistic updates for instant feedback',
      'Version vectors track causality between edits',
      'Google Docs uses Causal Consistency + OT in practice',
    ],
  },
  {
    id: 'consistency-models-disc-q2',
    question:
      'Your e-commerce platform uses DynamoDB with eventually consistent reads by default. The product team reports that users sometimes see old product prices even after updating them. Should you switch to strongly consistent reads? Discuss the trade-offs.',
    sampleAnswer: `This is a nuanced decision that depends on the specific requirements. My answer would be: **Use a hybrid approach** - eventually consistent reads for most cases, strongly consistent reads only when critical.

**Problem Analysis:**

The issue is **replication lag** in DynamoDB:
\`\`\`
Time T1: Admin updates price: $50 → $45 (write to primary)
Time T2: Replication to replicas (takes 100-500ms typically)
Time T3: User reads from replica → Sees $50 (stale)
Time T4: Eventually consistent → User sees $45
\`\`\`

**Should We Switch to Strongly Consistent Reads?**

**Short answer: No, not for everything.** Here\'s why:

**Trade-Offs of Strongly Consistent Reads:**

❌ **Double the cost**
- Eventually consistent reads: $0.25 per million reads
- Strongly consistent reads: $0.50 per million reads (2x)
- At 100M requests/day: $10/day vs $20/day = $3,650/year extra

❌ **Higher latency**
- Eventually consistent: 1-5ms (local replica)
- Strongly consistent: 10-20ms (must coordinate replicas)
- Page load time increases by 10-15ms

❌ **Lower availability**
- Eventually consistent: Reads from any replica (highly available)
- Strongly consistent: Requires quorum, fails if replicas unavailable

**When Stale Price is Actually a Problem:**

Let's evaluate the business impact:

**Scenario 1: Viewing Product Page**
\`\`\`
User browses product, sees $50 (actually $45)

Impact: User sees wrong price for 1-2 seconds
Business Risk: Low
- Price updates are rare (maybe once per day)
- Staleness typically <1 second in practice
- User might see $50, refresh, then see $45
- Slightly annoying but not broken

Decision: Eventually consistent OK
\`\`\`

**Scenario 2: Adding to Cart**
\`\`\`
User clicks "Add to Cart" after seeing $50
Backend calculates using current price $45

Impact: User added item expecting $50, cart shows $45
Business Risk: Medium
- Creates confusion
- But user sees correct price before checkout
- Can update quantity or remove if wrong

Decision: Eventual consistency acceptable with validation at checkout
\`\`\`

**Scenario 3: Checkout**
\`\`\`
User proceeds to checkout

Impact: THIS is where price must be accurate
Business Risk: High
- Cannot charge different price than shown at checkout
- Legal/trust issues if wrong

Decision: Must use strongly consistent read at checkout
\`\`\`

**My Recommended Solution:**

**Hybrid Consistency Strategy:**

\`\`\`
1. Product Browsing (Eventually Consistent):
   - Use eventually consistent reads
   - Low latency, low cost
   - Stale price for 1-2 seconds acceptable

2. Add to Cart (Eventually Consistent + Revalidation):
   - Initially use eventually consistent read (fast)
   - Show "Price subject to change at checkout"
   - Revalidate price at checkout (strongly consistent)

3. Checkout (Strongly Consistent):
   - Use strongly consistent read for final price
   - Ensure accuracy before payment
   - Show price mismatch warning if changed

4. Admin Price Updates (Read-Your-Writes):
   - Admin who updates price sees new price immediately
   - Use session consistency or route to same replica
\`\`\`

**Implementation:**

\`\`\`typescript
// Product page - eventually consistent (fast, cheap)
const product = await dynamoDB.get({
  TableName: 'Products',
  Key: { productId },
  ConsistentRead: false // Default, eventually consistent
});

// Checkout - strongly consistent (accurate)
const productAtCheckout = await dynamoDB.get({
  TableName: 'Products',
  Key: { productId },
  ConsistentRead: true // Strongly consistent
});

if (productAtCheckout.price !== cartItem.price) {
  // Warn user: "Price changed from $50 to $45"
  showPriceChangeWarning();
}
\`\`\`

**Additional Improvements:**

**1. Cache with Short TTL**
\`\`\`
Redis cache with 30-second TTL:
- Most product page views hit cache (sub-ms latency)
- Cache miss: Read from DynamoDB (eventually consistent)
- Price updates clear cache
- Staleness: Maximum 30 seconds (acceptable)
- Cost: Dramatically reduced DynamoDB reads
\`\`\`

**2. Optimistic UI with Validation**
\`\`\`
UI: Show price from cache/eventually consistent read
Backend: Validate at checkout with strongly consistent read
Result: Fast browsing, accurate checkout
\`\`\`

**3. Version-Based Pricing**
\`\`\`
Add version field to products:
- Product price changes → increment version
- Cart stores: productId + price + version
- Checkout validates: current version matches cart version
- If mismatch: Show "Price updated to $45"
\`\`\`

**Business Impact Analysis:**

**If we DON'T change anything:**
- Cost: $10/day (eventually consistent)
- Latency: 1-5ms per product page
- User Experience: Occasional 1-2 second stale price
- Risk: Low (validated at checkout)

**If we switch everything to strongly consistent:**
- Cost: $20/day (+$3,650/year)
- Latency: 10-20ms per product page (feels sluggish)
- User Experience: Always accurate, but slower
- Risk: None

**If we use hybrid approach:**
- Cost: ~$11/day (mostly eventual, strongly consistent only at checkout)
- Latency: 1-5ms browsing, 10-20ms only at checkout (acceptable)
- User Experience: Fast browsing, accurate checkout
- Risk: None

**Recommendation:**

**Use the hybrid approach:**
1. Keep eventually consistent reads for browsing (fast, cheap)
2. Add caching with short TTL (even faster)
3. Use strongly consistent reads only at checkout (accurate when it matters)
4. Show price change warnings if price differs from cart

**Key Message to Product Team:**

"The stale price issue affects 1-2 seconds during browsing, which has low business impact since we validate prices at checkout. Switching everything to strongly consistent would add 10-15ms latency to every product page, making the site feel slower, and double our database costs ($3,650/year). Instead, we should use strongly consistent reads only at checkout where accuracy is critical, keeping fast browsing experience while ensuring users are charged correctly."`,
    keyPoints: [
      "Don't use strongly consistent reads for everything - costs 2x and adds latency",
      'Hybrid approach: Eventually consistent for browsing, strongly consistent at checkout',
      'Business impact of stale data varies by use case',
      'Validate critical operations (checkout) with strong consistency',
      'Add caching layer to reduce latency and cost further',
      'Show price change warnings if price differs between cart and checkout',
      'Optimize for common case (browsing) while ensuring accuracy where it matters (checkout)',
    ],
  },
  {
    id: 'consistency-models-disc-q3',
    question:
      'Explain to a junior developer why their proposed "eventually consistent distributed counter" for a rate limiter won\'t work reliably. What consistency model does rate limiting actually require?',
    sampleAnswer: `Great question! Let me explain why eventual consistency breaks rate limiting and what consistency model is actually required.

**The Problem with Eventual Consistency for Rate Limiting:**

**Proposed Design (Broken):**
\`\`\`
Rate Limit: 100 requests per minute per user
Implementation: Eventually consistent counter in Cassandra

Node A: User makes request 1 → Increment counter (async replication)
Node B: User makes request 2 → Reads counter, sees 0 (stale!)
Node B: Allows request (counter = 1)
Node A: User makes request 3 → Reads counter, sees 1 (replication lag)
Node A: Allows request (counter = 2)
...
Result: User makes 200 requests before any node reaches 100

Problem: Eventually consistent counter doesn't provide real-time accurate count
\`\`\`

**Why This Breaks:**

**1. Replication Lag Allows Bypass**
\`\`\`
Time T1: User sends 50 requests to Node A
         Node A counter: 50
         
Time T2: User switches to Node B (different replica)
         Node B counter: 0 (replication not complete)
         Node B allows next 50 requests
         
Result: 100 requests sent, but should be rate limited at 100
        With 3 replicas, user could send 300 requests!
\`\`\`

**2. Concurrent Requests Race**
\`\`\`
User at 99 requests (just under limit)
Sends 10 concurrent requests to different nodes

Node A: Reads 99, allows request, increments to 100
Node B: Reads 99 (stale), allows request, increments to 100  
Node C: Reads 99 (stale), allows request, increments to 100
...
All 10 nodes read 99 and allow requests

Result: User sent 109 requests (9 over limit)
\`\`\`

**3. Malicious Actors Can Exploit**
\`\`\`
Attacker knows system uses eventual consistency
Sends requests to different replicas rapidly
Each replica has stale counter
Attacker bypasses rate limit by 3-5x
\`\`\`

**What Consistency Model Rate Limiting Requires:**

**Answer: Rate limiting requires Strong Consistency (or stronger, like Linearizability)**

**Why:**
- Must enforce global limit across all servers
- Cannot allow "cheating" by hitting different replicas
- Counter must be accurate in real-time
- Increments must be atomic and immediately visible

**Correct Implementation Options:**

**Option 1: Centralized Counter (Redis with Strong Consistency)**

\`\`\`
Architecture:
- Redis Cluster with strong consistency (wait for replication)
- All rate limit checks go to Redis
- INCR command is atomic

Code:
const count = await redis.incr(\`rate_limit:user:\${userId}:\${window}\`);
if (count === 1) {
  await redis.expire(\`rate_limit:user:\${userId}:\${window}\`, 60); // 1 minute TTL
}

if (count > 100) {
  return { allowed: false, message: "Rate limit exceeded" };
}
return { allowed: true };

Why it works:
- Redis INCR is atomic (linearizable)
- All servers see same counter value immediately
- No race conditions
\`\`\`

**Option 2: Distributed Rate Limiting with Consensus**

\`\`\`
Use etcd or ZooKeeper (both provide linearizability):
- Store counter in etcd
- Use compare-and-swap (CAS) for atomic increment
- Raft consensus ensures all nodes agree on count

Pseudo-code:
count = etcd.get(\`rate_limit:user:\${userId}\`);
if (count >= 100) reject();

success = etcd.cas(\`rate_limit:user:\${userId}\`, count, count + 1);
if (!success) retry(); // CAS failed, another request incremented

Why it works:
- Linearizable consistency
- CAS ensures atomic increments
- No two requests can both see count=99 and increment
\`\`\`

**Option 3: Token Bucket with Local Counters + Reservation**

\`\`\`
Hybrid approach for scale:

Central Authority (Redis):
- Allocates token "budgets" to each API server
- Example: User has 100 requests/min, allocated 20 to each of 5 servers

API Servers:
- Track local budget (20 requests)
- When budget exhausted, request more from central authority
- If central authority says "no budget left", reject requests

Why it works:
- Most requests use local counter (fast, no network)
- Central authority ensures global limit (strong consistency)
- Scale: Handles millions of requests with minimal central coordination
\`\`\`

**Option 4: Sliding Window with Redis Sorted Set**

\`\`\`typescript
// Accurate rate limiting using sorted set
async function checkRateLimit (userId: string): Promise<boolean> {
  const now = Date.now();
  const windowStart = now - 60000; // 60 seconds ago
  
  // Remove old entries (outside window)
  await redis.zremrangebyscore(
    \`rate_limit:\${userId}\`,
    '-inf',
    windowStart
  );
  
  // Count requests in current window
  const count = await redis.zcard(\`rate_limit:\${userId}\`);
  
  if (count >= 100) {
    return false; // Rate limit exceeded
  }
  
  // Add current request
  await redis.zadd(\`rate_limit:\${userId}\`, now, \`\${now}-\${uuid()}\`);
  return true;
}

Why it works:
- Sorted set operations are atomic
- Sliding window (not fixed window)
- Accurate count of requests in last 60 seconds
- Redis provides strong consistency
\`\`\`

**Comparison:**

| Approach | Consistency | Latency | Scale | Accuracy |
|----------|-------------|---------|-------|----------|
| **Eventual (Broken)** | Eventual | Low | High | ❌ Broken |
| **Redis Centralized** | Strong | Medium | Medium | ✓ Perfect |
| **etcd Consensus** | Linearizable | High | Low | ✓ Perfect |
| **Token Bucket Hybrid** | Strong (global) | Low (local) | High | ✓ Good enough |
| **Redis Sliding Window** | Strong | Medium | Medium | ✓ Perfect |

**What to Tell the Junior Developer:**

"Rate limiting requires **strong consistency** because we need an accurate, real-time count across all servers. Eventual consistency allows users to bypass the limit by sending requests to different replicas before replication completes. 

Imagine a user is at 99/100 requests. With eventual consistency, if they send 10 requests to different servers, each server might read the counter as 99 (stale) and allow the request, resulting in 109 requests total.

For rate limiting, use:
1. **Redis with atomic INCR** (simplest, works well for most cases)
2. **Token bucket with local counters** (if scale is critical)
3. **etcd/ZooKeeper** (if you need strongest guarantees)

Never use eventually consistent databases (Cassandra, DynamoDB) directly for rate limiting - they'll allow users to exceed limits."

**Key Insight:**

Rate limiting is a perfect example of where **correctness > performance**. Allowing users to bypass rate limits can lead to:
- DDoS vulnerability
- Resource exhaustion
- Unfair usage
- Revenue loss (if rate limit is tied to pricing tiers)

Therefore, strong consistency is non-negotiable, even if it costs a bit more latency.`,
    keyPoints: [
      'Eventually consistent counters allow bypassing rate limits via replication lag',
      'Rate limiting requires strong consistency (linearizability) for accurate enforcement',
      'Concurrent requests can race with eventual consistency, all seeing stale counts',
      'Correct implementations: Redis atomic INCR, etcd/ZooKeeper CAS, Token Bucket with central authority',
      'Token bucket hybrid approach scales well while maintaining global strong consistency',
      'Redis sorted set provides sliding window rate limiting with atomic operations',
      'Correctness more important than performance for security-critical features like rate limiting',
    ],
  },
];
