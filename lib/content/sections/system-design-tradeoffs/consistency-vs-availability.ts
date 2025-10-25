/**
 * Consistency vs Availability Section
 */

export const consistencyvsavailabilitySection = {
  id: 'consistency-vs-availability',
  title: 'Consistency vs Availability',
  content: `One of the most fundamental trade-offs in distributed systems is choosing between **consistency** and **availability**. This decision impacts every aspect of your architecture and is a common interview topic.

## Understanding the Trade-off

The **CAP theorem** (Consistency, Availability, Partition Tolerance) states that in the presence of a network partition, you must choose between consistency and availability.

**Consistency**: All nodes see the same data at the same time. A read always returns the most recent write.

**Availability**: Every request receives a response (success or failure), even if some nodes are down.

**Partition Tolerance**: The system continues to operate despite network failures between nodes.

**Reality**: Network partitions WILL happen (hardware failures, network issues). So in practice, you're choosing between CP (Consistency + Partition Tolerance) or AP (Availability + Partition Tolerance).

---

## Consistency-First (CP) Systems

**Choose consistency when**:
- Data accuracy is critical
- Stale data causes problems
- Financial transactions
- Inventory management
- User account balances

### Example: Banking System

**Scenario**: You have $100 in your account. You withdraw $100 at ATM A. Simultaneously, your spouse tries to withdraw $100 at ATM B.

**CP System (Consistent)**:
1. ATM A locks your account, withdraws $100, balance = $0
2. ATM B tries to withdraw, sees $0, transaction denied ✅
3. Result: One withdrawal succeeds, one fails (correct)

**AP System (Available)**:
1. Network partition between datacenters
2. Both ATMs see balance = $100 (stale data)
3. Both withdrawals succeed 💰💰
4. Result: Overdraft by $100 (incorrect) ❌

**For banking, consistency is non-negotiable.** It\'s better to deny service (unavailable) than to show incorrect balance.

### CP Databases
- **PostgreSQL**: Single-master, strong consistency
- **MongoDB** (default): Strong consistency with primary reads
- **HBase**: Strong consistency via single RegionServer
- **Redis** (single instance): Strong consistency
- **Consul**: CP for service discovery

---

## Availability-First (AP) Systems

**Choose availability when**:
- System must always respond
- Stale data is acceptable
- User experience > perfect accuracy
- Social media feeds
- Product catalogs
- Analytics dashboards

### Example: Facebook News Feed

**Scenario**: You post "I got engaged! 💍". It takes 2 seconds to replicate to all datacenters.

**AP System (Available)**:
1. You post from San Francisco datacenter
2. Your friend in Tokyo refreshes immediately
3. Tokyo datacenter shows stale feed (no engagement post yet)
4. 2 seconds later, post appears
5. Result: Slightly delayed, but feed always loads ✅

**CP System (Consistent)**:
1. San Francisco datacenter accepts post
2. Must replicate to all datacenters before confirming
3. If Tokyo datacenter is down, post fails ❌
4. Result: Post rejected due to datacenter outage

**For social media, availability matters more.** Users prefer a slightly stale feed over "Service Unavailable" errors.

### AP Databases
- **Cassandra**: Eventually consistent, highly available
- **DynamoDB**: Tunable consistency (eventual by default)
- **Riak**: Designed for availability
- **Couchbase**: AP system
- **Cosmos DB**: Tunable, often AP

---

## Real-World Examples

### 1. Amazon Shopping Cart (AP)

**Decision**: Availability

**Why**: Better to add item to cart and merge later than to fail the add-to-cart request. Amazon's famous "shopping cart divergence" paper describes how they prioritize writes being accepted, even if it means merging carts later.

**Result**: You might briefly see different cart contents on web vs mobile, but items are never lost.

### 2. Google Docs Collaboration (AP)

**Decision**: Availability (with eventual consistency)

**Why**: Multiple users editing the same document. Each user's edits should be accepted immediately, even if they conflict.

**Result**: Operational Transformation (OT) or CRDTs resolve conflicts. You might see brief inconsistencies, but edits are never lost.

### 3. Stripe Payment Processing (CP)

**Decision**: Consistency

**Why**: Payment must be processed exactly once. Double-charging is worse than a failed payment.

**Result**: If there's any doubt about payment state, the request fails and user retries.

### 4. Instagram Likes (AP)

**Decision**: Availability

**Why**: Like counts don't need to be perfectly accurate in real-time. A post showing 1,234 likes vs 1,237 likes is fine.

**Result**: Like counts might briefly differ across regions, but feed always loads.

### 5. Ticket Booking (CP)

**Decision**: Consistency

**Why**: Can't sell the same seat twice. Inventory must be accurate.

**Result**: During high traffic (Taylor Swift concert), some users see "sold out" even if seats might become available moments later.

---

## Hybrid Approaches

Most real systems use **different consistency models for different data**:

### Example: E-commerce System

**Strong Consistency (CP)**:
- Inventory count
- Payment processing
- Order status

**Eventual Consistency (AP)**:
- Product reviews
- Product recommendations
- Wish lists
- Browse history

**Why**: Selling the same item twice is bad (consistency). But failing to load product reviews is also bad (availability). Use CP for critical data, AP for less critical.

---

## Tunable Consistency

Modern databases offer **tunable consistency** where you choose per request:

### Cassandra Example

**Write**: W = number of replicas that must acknowledge
**Read**: R = number of replicas to query

**Strong consistency**: W + R > N (N = total replicas)
- Example: N=3, W=2, R=2 (2+2 > 3) → Strong consistency
- Trade-off: Higher latency (must wait for 2 nodes)

**Eventual consistency**: W=1, R=1
- Trade-off: Faster (only 1 node) but might read stale data

**Use case**:
- Critical writes (payment): W=QUORUM (majority)
- Non-critical reads (browse products): R=ONE (fastest)

---

## Interview Discussion Framework

When asked "Should this system prioritize consistency or availability?", use this framework:

### 1. Clarify Data Types

**Question**: "What data are we storing? Are we dealing with financial transactions, user-generated content, or analytics?"

### 2. Identify Critical vs Non-Critical

**Critical** (lean CP):
- Money, inventory, user authentication
- "Would incorrect data cause legal/financial issues?"

**Non-Critical** (lean AP):
- Social content, analytics, caches
- "Is slightly stale data acceptable?"

### 3. Discuss User Experience

**Question**: "Is it worse to show stale data or to show an error message?"

**Example**: 
- Banking: Error better than wrong balance (CP)
- Twitter: Stale tweet better than error (AP)

### 4. Consider Hybrid

**Suggestion**: "We could use strong consistency for orders but eventual consistency for product catalogs."

---

## Common Mistakes

### ❌ Mistake 1: "Just Use Strong Consistency Everywhere"

**Problem**: Strong consistency has costs:
- Higher latency (multi-datacenter coordination)
- Lower availability (single point of failure)
- Reduced throughput (coordination overhead)

**Reality**: Most data doesn't need strong consistency.

### ❌ Mistake 2: "CAP Means Choose 2 of 3"

**Problem**: Misunderstanding CAP. You MUST have P (partition tolerance) in distributed systems.

**Correct**: Choose between C or A when partitions occur.

### ❌ Mistake 3: "Eventual Consistency = Broken"

**Problem**: Eventual consistency is not "no consistency." It means "all nodes will converge to same state eventually."

**Reality**: Most systems use eventual consistency successfully (Amazon, Facebook, etc.).

### ❌ Mistake 4: "One Size Fits All"

**Problem**: Treating all data the same way.

**Better**: Different consistency models for different data types in the same system.

---

## Best Practices

### ✅ 1. Default to Eventual Consistency, Upgrade When Needed

Start with AP (eventual consistency), use CP only where truly needed. Most data doesn't require strong consistency.

### ✅ 2. Use Idempotency Keys

Even in eventually consistent systems, make operations idempotent so repeated operations are safe.

### ✅ 3. Design for Partition Tolerance

Network partitions WILL happen. Design systems to handle them gracefully.

### ✅ 4. Communicate Consistency Model to Users

If eventual consistency means user sees stale data, design UI to indicate this:
- "Syncing..."
- "Last updated 5 seconds ago"
- Optimistic UI updates

### ✅ 5. Test Partition Scenarios

Use chaos engineering to test how system behaves during network partitions.

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd recommend an **AP approach with tunable consistency** for the following reasons:

**Core data (orders, payments)**: Strong consistency (CP)
- Use database transactions
- Accept higher latency (~100-200ms)
- Better to fail than to corrupt data

**Read-heavy data (product catalog)**: Eventual consistency (AP)
- Replicate globally
- Low latency (~10-20ms)
- Stale data acceptable (product description updates eventually)

**Justification**: 99% of requests are reads (browsing products), only 1% are writes (placing orders). Optimizing reads for availability provides better UX while maintaining strong consistency where it matters."

### Key Phrases to Use

- "Trade-off between..."
- "For this use case, I'd prioritize X because..."
- "The cost of strong consistency here is..."
- "Eventual consistency is acceptable because..."
- "We could use a hybrid approach..."

---

## Summary Table

| Aspect | Consistency (CP) | Availability (AP) |
|--------|-----------------|-------------------|
| **Priority** | Correct data | Always respond |
| **When Partition** | Deny requests | Accept stale reads |
| **Use Cases** | Banking, Inventory | Social, Catalogs |
| **Databases** | PostgreSQL, MongoDB | Cassandra, DynamoDB |
| **Latency** | Higher (coordination) | Lower (local reads) |
| **Throughput** | Lower | Higher |
| **User Experience** | Occasional errors | Always fast |
| **Example Error** | "Service unavailable" | Stale data shown |

---

## Key Takeaways

✅ CAP theorem: Choose C or A during network partitions (P is required)
✅ CP (Consistent): Banking, payments, inventory—accuracy critical
✅ AP (Available): Social media, catalogs, analytics—availability critical
✅ Most systems use **hybrid**: CP for critical data, AP for rest
✅ Tunable consistency (Cassandra, DynamoDB) allows per-request choice
✅ Eventual consistency is not broken—it powers Amazon, Facebook, etc.
✅ Design UI to communicate consistency model to users
✅ In interviews, justify your choice based on use case, don't just pick one`,
};
