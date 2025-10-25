/**
 * Strong Consistency vs Eventual Consistency Section
 */

export const strongvseventualconsistencySection = {
  id: 'strong-vs-eventual-consistency',
  title: 'Strong Consistency vs Eventual Consistency',
  content: `The consistency model you choose fundamentally shapes your system's behavior, performance, and complexity. This is one of the most important trade-offs in distributed systems.

## Definitions

**Strong Consistency** (also called **Linearizability** or **Immediate Consistency**):
- After a write completes, **all subsequent reads** see that write
- All nodes see the same data at the same time
- System behaves as if there's only one copy of data

**Eventual Consistency**:
- After a write, reads **may return stale data** temporarily
- All nodes will **eventually** converge to the same value
- No guarantee on how long "eventually" takes (could be milliseconds or seconds)

---

## Strong Consistency in Action

### Example: Bank Account Transfer

**Scenario**: You transfer $100 from Savings to Checking

**Strong Consistency (PostgreSQL)**:

\`\`\`
Time  | Savings | Checking | What Happens
------|---------|----------|-------------
T0    | $1000   | $500     | Initial state
T1    | ---     | ---      | Transaction starts (both locked)
T2    | $900    | $600     | Transaction commits atomically
T3    | $900    | $600     | All reads see new values
\`\`\`

**Guarantees**:
- ✅ No read ever sees $1000 Savings + $500 Checking after T2 (lost money)
- ✅ No read ever sees $900 Savings + $500 Checking (partial update)
- ✅ All users see the same balances at the same time

**How it works**:
1. Write-ahead log (WAL) records transaction
2. Lock both accounts (prevent concurrent modifications)
3. Update both accounts atomically
4. Unlock accounts
5. All reads see new values

**Cost**: Lower availability (locks), higher latency (coordination)

---

## Eventual Consistency in Action

### Example: Facebook "Like" Count

**Scenario**: You like a post that has 1,234 likes

**Eventual Consistency (Cassandra)**:

\`\`\`
Time  | US-East | EU-West | Asia   | What Happens
------|---------|---------|--------|-------------
T0    | 1234    | 1234    | 1234   | Initial state (replicated)
T1    | 1235    | 1234    | 1234   | Your like recorded in US-East
T2    | 1235    | 1235    | 1234   | Replicated to EU-West (async)
T3    | 1235    | 1235    | 1235   | Replicated to Asia (async)
T4+   | 1235    | 1235    | 1235   | Eventually consistent!
\`\`\`

**Reality**:
- ✅ Your like is recorded immediately (fast!)
- ⚠️ Different users see different like counts temporarily
- ✅ Eventually, all users see 1,235 likes
- ✅ System remains available even if Asia datacenter goes down

**How it works**:
1. Write accepted at nearest datacenter (US-East)
2. Write confirmed immediately (don't wait for other datacenters)
3. Background replication to other datacenters
4. Conflict resolution if concurrent writes

**Cost**: Temporary inconsistency (different users see different counts)

---

## Consistency Spectrum

Consistency is not binary. There\'s a spectrum:

\`\`\`
Strong Consistency ◄────────────────────────► Eventual Consistency
      |                    |                         |
  Linearizable        Causal                    Eventual
  (Strongest)         Consistency               (Weakest)
      |                    |                         |
  PostgreSQL          MongoDB                  Cassandra
  Redis               DynamoDB                 DynamoDB
  (single-master)     (strong option)          (eventual option)
\`\`\`

### Consistency Models Explained

**1. Linearizable (Strong Consistency)**:
- Reads always see the most recent write
- Total global order of operations
- Example: Single-master database (PostgreSQL, MySQL)

**2. Sequential Consistency**:
- All processes see operations in the same order
- Order may differ from real-time order
- Example: Coordinated updates with version numbers

**3. Causal Consistency**:
- Operations with causal relationships are seen in order
- Concurrent operations may be seen in different orders
- Example: Comment on a post (causally related) vs two unrelated likes (concurrent)

**4. Eventual Consistency**:
- No ordering guarantees during inconsistency window
- All replicas converge eventually
- Example: Multi-master databases (Cassandra, DynamoDB with eventual reads)

---

## When to Use Strong Consistency

### Use Cases Requiring Strong Consistency

**1. Financial Transactions**
- Bank transfers, payment processing
- Account balances
- **Why**: Money must be accurate. Can't have lost or duplicate money.

**2. Inventory Management**
- Product stock counts
- Ticket/seat booking
- **Why**: Can't sell what you don't have. Overselling is bad.

**3. User Authentication**
- Password changes, account status
- Security permissions
- **Why**: Security-critical. Must see latest password immediately.

**4. Booking Systems**
- Hotel reservations, flight booking
- **Why**: Can't double-book the same resource.

**5. Auction Systems**
- Current highest bid
- **Why**: Bid must be accurate for fairness.

### Example: Ticket Booking (Ticketmaster)

**Problem**: Taylor Swift concert, 10,000 people trying to book 10 seats

**With Strong Consistency**:
\`\`\`
Transaction 1: Check seat A1 (available) → Lock → Book → Commit ✅
Transaction 2: Check seat A1 (locked, wait...) → Unavailable ❌
\`\`\`

**Without Strong Consistency (Eventual)**:
\`\`\`
Transaction 1: Check seat A1 (available) → Book → Async replication
Transaction 2: Check seat A1 (available, replication lag) → Book
Result: Seat A1 booked twice! 💥
\`\`\`

**Solution**: Use strong consistency for inventory. Accept lower throughput.

---

## When to Use Eventual Consistency

### Use Cases Accepting Eventual Consistency

**1. Social Media Feeds**
- Twitter timeline, Facebook news feed
- **Why**: Slightly stale posts acceptable. UX tolerates eventual consistency.

**2. Analytics and Dashboards**
- Page view counts, aggregated metrics
- **Why**: Exact real-time numbers not critical. Approximate is fine.

**3. Product Catalogs**
- E-commerce product listings, descriptions
- **Why**: Product info changes infrequently. Stale data for seconds is acceptable.

**4. User-Generated Content**
- Comments, reviews, forum posts
- **Why**: Users tolerate brief delay before content visible globally.

**5. CDN-Cached Content**
- Cached web pages, images
- **Why**: Stale cached content acceptable for performance.

### Example: Twitter Timeline

**Problem**: User tweets, 10M followers should see it

**With Strong Consistency**:
\`\`\`
User tweets → Wait for replication to ALL datacenters → Confirm
Latency: 500ms - 2 seconds (global coordination)
If any datacenter down, tweet fails ❌
\`\`\`

**With Eventual Consistency**:
\`\`\`
User tweets → Write to local datacenter → Confirm immediately
Latency: 50ms ⚡
Async replication to other datacenters (2-5 seconds)
Followers see tweet within seconds (eventual)
System available even if datacenter down ✅
\`\`\`

**Result**: Eventual consistency enables high availability and low latency for non-critical social content.

---

## The Read-Your-Writes Guarantee

A special case: **Read-Your-Writes Consistency**

**Problem**: User posts content, immediately refreshes, doesn't see their own post!

**Bad UX**:
\`\`\`
User: Posts photo to Instagram
Writes to US-East datacenter
User: Refreshes feed immediately
Reads from EU-West datacenter (not replicated yet)
User: "Where\'s my photo?! App is broken!" 😡
\`\`\`

**Solution - Read-Your-Writes**:
\`\`\`
User: Posts photo (writes to US-East)
Cookie stores: "last_write_datacenter=US-East"
User: Refreshes feed
App routes read to US-East (where write happened)
User: Sees own photo immediately ✅
\`\`\`

**Implementation**:
- Track where user's last write went
- Route subsequent reads to same datacenter
- After replication completes, resume normal routing

**Benefits**: Combines eventual consistency (system-wide) with strong consistency (user's own view)

---

## Conflict Resolution in Eventual Consistency

**Problem**: Two concurrent writes to the same data with eventual consistency

### Example: Collaborative Document Editing

**Scenario**: Alice and Bob both edit the same document offline

\`\`\`
Initial: "Hello World"
Alice edits (offline): "Hello Beautiful World"
Bob edits (offline): "Hello Amazing World"
Both come online simultaneously
\`\`\`

**What should final state be?**

### Resolution Strategies

**1. Last-Write-Wins (LWW)**
- Use timestamp, keep the later write
- Simple but loses data
- Example: Bob\'s write at 10:01 wins over Alice's at 10:00
- **Problem**: Alice's edit lost! ❌

**2. Version Vectors**
- Track version history per node
- Detect conflicts, keep both versions
- Application resolves conflict
- **Example**: Git merge conflicts

**3. CRDTs (Conflict-free Replicated Data Types)**
- Special data structures that merge automatically
- Example: Counter CRDT (merge by summing)
- **Use case**: Google Docs uses similar technique

**4. Application-Level Resolution**
- Application decides how to merge
- Example: Shopping cart (merge both items)
- Example: Social media (show both versions, user picks)

### Example: Shopping Cart (Amazon)

**Scenario**: User adds Item A on laptop, Item B on mobile (both offline)

**Last-Write-Wins**: Would lose one item ❌

**Application-Level Merge**:
\`\`\`
Cart on laptop: [Item A]
Cart on mobile: [Item B]
Both sync
Merged cart: [Item A, Item B] ✅
\`\`\`

**Amazon\'s choice**: Always merge carts (never lose items). Better to have duplicate than lose customer's item.

---

## Performance Implications

### Strong Consistency

**Advantages**:
- ✅ Simple application logic (always see latest data)
- ✅ No conflict resolution needed
- ✅ Easier to reason about

**Disadvantages**:
- ❌ Higher latency (coordination overhead)
- ❌ Lower availability (coordination can fail)
- ❌ Lower throughput (locking, coordination)
- ❌ Scalability limits (coordination overhead grows)

**Latency example**:
- Single datacenter strong read: 5-10ms
- Multi-datacenter strong read: 100-500ms (cross-region coordination)

### Eventual Consistency

**Advantages**:
- ✅ Lower latency (no coordination)
- ✅ Higher availability (accept writes even if some nodes down)
- ✅ Higher throughput (parallel writes)
- ✅ Better scalability (less coordination)

**Disadvantages**:
- ❌ Complex application logic (handle stale data)
- ❌ Conflict resolution needed
- ❌ Harder to reason about
- ❌ Inconsistency window (unpredictable)

**Latency example**:
- Local write: 5-10ms (immediate)
- Global read: 5-10ms (may be stale)
- Convergence time: 100ms - 5 seconds (depends on system)

---

## Hybrid Approaches

Most real systems use **different consistency models for different data**.

### Example: E-commerce Platform (Amazon)

**Strong Consistency**:
- Order status (pending → confirmed → shipped)
- Payment processing
- Inventory (during checkout)

**Eventual Consistency**:
- Product catalog (descriptions, images)
- Product reviews
- Recommendations
- Browse history

**Read-Your-Writes**:
- User's own orders (always see your orders)
- User\'s own reviews (see your review immediately)

**Why**: 99% of traffic is browsing (eventual consistency, fast). 1% is checkout (strong consistency, correct).

---

## Common Mistakes

### ❌ Mistake 1: "Eventual Consistency = Data Loss"

**Wrong**: "Eventual consistency means data gets lost sometimes."

**Correct**: Eventual consistency means temporary stale reads, but writes are durable. All nodes converge to same state eventually.

### ❌ Mistake 2: "Strong Consistency Always Better"

**Wrong**: "Always use strong consistency to be safe."

**Correct**: Strong consistency has costs (latency, availability). Use only where needed.

### ❌ Mistake 3: "Eventual Consistency Means No Guarantees"

**Wrong**: "Eventual consistency has no guarantees."

**Correct**: Eventual consistency guarantees convergence. With proper conflict resolution, data is not lost.

### ❌ Mistake 4: "Same Consistency for All Data"

**Wrong**: "Pick one consistency model for entire system."

**Correct**: Use different models for different data types based on requirements.

---

## Best Practices

### ✅ 1. Default to Eventual, Upgrade When Needed

Start with eventual consistency (faster, more scalable). Use strong consistency only for critical data.

### ✅ 2. Communicate Inconsistency to Users

If using eventual consistency, design UI to handle stale data:
- "Updating..."
- "Last synced 2 seconds ago"
- Optimistic UI (show immediately, sync later)

### ✅ 3. Use Idempotency

Make operations idempotent so retries/duplicates are safe in eventually consistent systems.

### ✅ 4. Test for Inconsistency

Chaos engineering: Introduce replication delays, test how app behaves with stale data.

### ✅ 5. Choose Appropriate Conflict Resolution

Pick conflict resolution strategy based on data type:
- Counters: Sum (CRDT)
- Shopping cart: Merge (application-level)
- User settings: Last-write-wins (with version numbers)

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd recommend **[strong/eventual/hybrid]** consistency because:

**Data criticality**: [Financial? Social content?]

**Strong consistency for**:
- [List critical data like payments, inventory]
- Implementation: Single-master database, transactions
- Trade-off: Higher latency (~100ms) but correctness guaranteed

**Eventual consistency for**:
- [List non-critical data like catalogs, reviews]
- Implementation: Multi-master, async replication
- Trade-off: Stale reads briefly, but fast (~10ms) and highly available

**User experience**: [How will users perceive inconsistency?]

**Conflict resolution**: [If eventual, how will conflicts be resolved?]"

### Example: Design Instagram

"Instagram needs a hybrid consistency model:

**Strong consistency**:
- User authentication (password changes must be immediate)
- Account status (blocked users can't post)

**Eventual consistency**:
- Photo feed (slightly stale feed acceptable)
- Like counts (exact count not critical in real-time)

**Read-your-writes**:
- User's own posts (must see own photo immediately after posting)

**Conflict resolution**:
- Likes: Sum (CRDT counter)
- Follow relationships: Last-write-wins with timestamp

**Result**: System is highly available and fast for browsing (99% of traffic) while maintaining correctness for critical operations (login, posting)."

---

## Summary Table

| Aspect | Strong Consistency | Eventual Consistency |
|--------|-------------------|---------------------|
| **Guarantee** | Read = latest write | Reads eventually consistent |
| **Latency** | Higher (coordination) | Lower (local ops) |
| **Availability** | Lower (coordination can fail) | Higher (accepts stale) |
| **Throughput** | Lower (locking) | Higher (parallel) |
| **Complexity** | Simpler app logic | Complex conflict resolution |
| **Use Cases** | Banking, inventory | Social media, catalogs |
| **Databases** | PostgreSQL, MySQL | Cassandra, DynamoDB |
| **Conflicts** | Prevented (locks) | Must be resolved |
| **User Impact** | Occasional errors | Stale data briefly |

---

## Key Takeaways

✅ Strong consistency: All nodes see same data immediately (higher cost)
✅ Eventual consistency: Nodes converge eventually (lower cost, may see stale)
✅ Strong for: Banking, inventory, authentication (correctness critical)
✅ Eventual for: Social feeds, analytics, catalogs (speed > perfect accuracy)
✅ Read-your-writes: User sees their own changes immediately (best UX)
✅ Most systems use hybrid: Different consistency for different data
✅ Conflict resolution required for eventual consistency (LWW, CRDTs, merge)
✅ In interviews, justify choice based on data criticality and user expectations`,
};
