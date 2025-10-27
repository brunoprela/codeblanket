/**
 * Quiz questions for CAP Theorem Deep Dive section
 */

export const captheoremQuiz = [
  {
    id: 'cap-theorem-disc-q1',
    question:
      'Your e-commerce platform uses PostgreSQL (CP) for inventory to prevent overselling. During Black Friday, the database replica fails, causing the system to reject new orders for 5 minutes. Leadership is upset about lost revenue and wants to switch to an AP system (Cassandra) to maintain availability. How would you respond? Discuss the trade-offs.',
    sampleAnswer: `I would **strongly advise against switching to an AP system** for inventory management, as this would risk overselling products - a worse problem than temporary unavailability. However, I would propose a **hybrid architecture** that maintains CP for inventory while improving availability.

**Why AP is Wrong for Inventory:**

**Problem with AP (Eventual Consistency):**
\`\`\`
Scenario: 1 item left in stock
- User A in US datacenter buys item → Success
- User B in EU datacenter (partitioned) buys item → Success
- Both purchases succeed due to AP
- Result: Sold 2 items when we only had 1 (oversold)
\`\`\`

**Business Impact of Overselling:**1. **Customer dissatisfaction**: Canceling orders after payment
2. **Reputation damage**: "Retailer oversells products"
3. **Legal issues**: Advertising products you can't deliver
4. **Lost trust**: Customers won't trust future purchases

**This is worse than 5 minutes of downtime.**

**Why 5 Minutes Downtime Happened:**

The issue isn't PostgreSQL being CP - it's **lack of high availability setup**:
- Single replica failure shouldn't cause 5 minute outage
- Need automatic failover (<30 second downtime)
- Need multiple replicas for redundancy

**My Proposed Solution:**

**1. Keep PostgreSQL (CP) for Inventory**
\`\`\`
Why: Correctness is critical
- Cannot oversell products
- Inventory must be accurate
- Better to show "temporarily unavailable" than sell items we don't have
\`\`\`

**2. Improve High Availability (HA)**
\`\`\`
Current: Primary + 1 Replica
Problem: Single replica failure = downtime

Improved: Primary + 3 Replicas + Auto-Failover
- Primary in US-East-1a
- Replica 1 in US-East-1b (same region, different AZ)
- Replica 2 in US-East-1c
- Replica 3 in US-West-1 (different region)

Use: Patroni or AWS RDS Multi-AZ
- Automatic failover in <30 seconds
- Health checks every 10 seconds
- Promotes replica to primary automatically
\`\`\`

**3. Implement Fallback Strategies**
\`\`\`
During temporary inventory DB unavailability:

Option A: Cached Inventory (Read-Only Mode)
- Show products with last known inventory
- Disable "Add to Cart" button with message: "Temporarily unavailable, try again in 1 minute"
- Better UX than complete outage

Option B: "Request to Purchase"
- Accept purchase requests with caveat: "Order processing, will confirm availability in 5 minutes"
- Validate inventory when DB recovers
- Cancel orders if oversold, apologize with discount code

Option C: Optimistic Inventory
- Allow purchases with small buffer (if inventory = 100, allow 105)
- Accept 5% risk of overselling
- Manually fulfill or cancel excess
\`\`\`

**4. Hybrid Architecture**

\`\`\`
PostgreSQL (CP):
- Inventory (must be accurate)
- Orders (ACID transactions required)
- Payments (must not double-charge)

Redis (AP):
- Product catalog cache (stale data acceptable)
- Session storage
- Shopping cart (can lose, user will re-add items)

Cassandra (AP):
- Product reviews (eventual consistency fine)
- User activity logs
- Search history
\`\`\`

**Trade-offs Analysis:**

**Option 1: Keep PostgreSQL (CP) + Improve HA**
- ✅ Inventory always accurate (no overselling)
- ✅ Downtime reduced to <30 seconds (auto-failover)
- ✅ Customer trust maintained
- ❌ During partition, system may be unavailable (<1 min)
- ❌ More expensive (3+ replicas, failover infrastructure)

**Option 2: Switch to Cassandra (AP)**
- ✅ Always available during partitions
- ✅ Scales horizontally easily
- ❌ **Risk of overselling** (unacceptable for business)
- ❌ Complex conflict resolution logic
- ❌ Customer complaints and reputation damage
- ❌ Massive engineering effort to rewrite inventory system

**Recommendation:**

**Keep CP, improve availability:**1. **Implement HA setup** (primary + 3 replicas, auto-failover) → Reduces downtime from 5 minutes to <30 seconds
2. **Add fallback UX** (cached product pages in read-only mode) → Degraded but functional
3. **Use AP for non-critical data** (product catalog, reviews) → Offload PostgreSQL

**Cost-Benefit:**
- **HA infrastructure cost**: $500/month additional
- **Engineering effort**: 2 weeks to implement
- **Revenue protected**: No overselling, customer trust maintained
- **Downtime**: 5 minutes → 30 seconds

**Key Message to Leadership:**

"The 5 minutes of downtime is a **high availability problem**, not a consistency problem. Switching to AP would trade rare downtime for frequent overselling - a worse problem. I recommend investing in HA infrastructure (auto-failover, multiple replicas) which solves the downtime issue while maintaining inventory accuracy."`,
    keyPoints: [
      'Inventory management requires CP (strong consistency) to prevent overselling',
      'Overselling damages customer trust and reputation worse than brief downtime',
      'The problem is lack of HA setup, not PostgreSQL being CP',
      'Solution: Keep CP, add high availability (multiple replicas, auto-failover)',
      'Reduce downtime from 5 minutes to <30 seconds with proper HA',
      'Use fallback strategies (cached read-only mode) during outages',
      'Hybrid architecture: CP for critical (inventory), AP for non-critical (catalog)',
    ],
  },
  {
    id: 'cap-theorem-disc-q2',
    question:
      'Explain CAP theorem to a non-technical executive who wants to know why our distributed database can\'t be "consistent, available, AND partition tolerant." They argue that Amazon and Google manage to have all three, so we should too.',
    sampleAnswer: `Great question! Let me explain CAP theorem using a simple analogy, then address the Amazon/Google point.

**The Simple Analogy:**

Imagine you have **two bank branches** (one in New York, one in California) that need to stay synchronized:

**Your bank account:**
- New York branch: balance = $100
- California branch: balance = $100

**Scenario: The phone lines between branches go down (network partition)**

You try to withdraw $80 in New York. What should happen?

**Option 1: Prioritize Consistency (CP)**
\`\`\`
New York branch: "I need to confirm with California branch that your balance is still $100 before allowing this withdrawal."
*tries to call California*
*phone line is down*
New York branch: "Sorry, I cannot process your withdrawal right now. Please try again later."
\`\`\`

**Result:** Your withdrawal is **rejected** (unavailable), but both branches will have the **same balance** when phone lines are restored (consistent).

**Option 2: Prioritize Availability (AP)**
\`\`\`
New York branch: "I'll allow your withdrawal even though I can't confirm with California. You have $100 in my records."
*gives you $80*

Meanwhile, you call California branch:
California branch: "Your balance is $100, how can I help?"
You: "Can I withdraw $80?"
California branch: "Sure!" *gives you $80*
\`\`\`

**Result:** Both branches **served you** (available), but now they have **different balances** (inconsistent):
- New York: $20
- California: $20
- **Problem:** You withdrew $160 from a $100 balance!

**The CAP Trade-off:**

When phone lines are down (partition), you **MUST choose**:
- **Be unavailable** (reject transactions until phone lines restore) ← **CP**
- **Allow inconsistency** (branches have different data temporarily) ← **AP**

**You cannot have both** during the phone outage because New York literally cannot communicate with California to maintain consistency while serving requests.

**Addressing "But Amazon and Google do it":**

They don't escape CAP theorem - they make **smart trade-offs** based on what data needs what guarantees:

**Amazon\'s Approach (Polyglot Persistence):**

\`\`\`
Payments & Inventory (CP):
- Use PostgreSQL or Aurora
- Strong consistency required
- If partition occurs, reject transactions
- Why: Cannot double-charge or oversell

Product Catalog (AP):
- Use DynamoDB
- Eventual consistency acceptable
- If partition occurs, show slightly stale data
- Why: Showing product description 1 second stale is fine

Shopping Cart (AP):
- Use DynamoDB
- Eventual consistency acceptable  
- Why: If user adds item and it takes 1 second to sync, no big deal
\`\`\`

**Google's Approach:**

\`\`\`
Gmail (AP):
- Uses BigTable/Spanner
- Your inbox might be slightly delayed (seconds)
- But always available
- Why: Better to see emails 1 second late than "email unavailable"

Google Ads Billing (CP):
- Uses Spanner with strong consistency
- Better to reject ad impression than mischarge advertiser
- Why: Money must be accurate
\`\`\`

**Key Insight:**

Amazon and Google **don't have all three** - they use **different databases with different CAP trade-offs for different data**:
- **Critical financial data** → CP (consistent, might be unavailable during partition)
- **User-facing content** → AP (available, might be slightly stale)

**What This Means for Our System:**

We should do what Amazon/Google do - **choose the right trade-off for each type of data**:

\`\`\`
Our System:

CP (Consistency + Partition Tolerance):
✓ Financial transactions
✓ Inventory management
✓ User authentication
→ Use PostgreSQL

AP (Availability + Partition Tolerance):
✓ User profiles
✓ Product recommendations
✓ Activity feeds
→ Use Cassandra/DynamoDB

Result: Right tool for each job
\`\`\`

**Bottom Line:**

CAP theorem is a **fundamental law of distributed systems**, like gravity. Amazon and Google don't escape it - they:
1. **Accept the trade-off** for each use case
2. **Use multiple databases** (some CP, some AP)
3. **Design systems** to minimize impact of partitions
4. **Optimize** for 99.9% of the time when there's no partition

We should do the same: use CP where consistency is critical (payments, inventory) and AP where availability matters more (product pages, recommendations).

**Analogy Summary:**

Just like you can't have a car that's simultaneously:
- Fast (high performance)
- Cheap (low cost)
- Reliable (never breaks)

You must **pick two**. Similarly with distributed databases:
- Consistent
- Available (during partitions)
- Partition tolerant

You must **pick two**, and since partitions will happen (networks fail), you must choose: **Consistent (CP)** or **Available (AP)** during those partitions.`,
    keyPoints: [
      "CAP theorem is a fundamental law - even Amazon/Google can't escape it",
      'Network partitions WILL happen - partition tolerance is mandatory',
      'Real choice: Consistency (CP) vs Availability (AP) during partitions',
      'Amazon/Google use different databases for different data types',
      'Critical financial data → CP (PostgreSQL, Spanner)',
      'User-facing content → AP (DynamoDB, BigTable)',
      'Solution: Polyglot persistence - right database for each use case',
    ],
  },
  {
    id: 'cap-theorem-disc-q3',
    question:
      'You are using Cassandra for your application. Someone claims Cassandra is "AP" but you notice that when you use consistency level QUORUM for reads and writes, the system sometimes becomes unavailable when nodes are down. Is Cassandra AP or CP? Explain the nuance.',
    sampleAnswer: `Cassandra is **neither strictly AP nor strictly CP** - it's **tunable**, allowing you to choose between CP and AP behavior **per query**. The classification depends on the consistency level you choose.

**Understanding Consistency Levels:**

Cassandra has configurable consistency levels that determine how many replica nodes must respond:

\`\`\`
Consistency Levels:
- ONE: Only 1 replica must respond
- QUORUM: Majority of replicas must respond (N/2 + 1)
- ALL: All replicas must respond
\`\`\`

**With Replication Factor 3:**

**Scenario 1: Consistency Level ONE (AP Behavior)**

\`\`\`
Nodes: A, B, C (replication factor = 3)
Node C fails or is partitioned

Write operation (CL=ONE):
- Write to Node A → Success
- Cassandra acknowledges write immediately
- Replicates to Node B asynchronously
- Node C is down (doesn't matter)
- ✅ Write succeeds (available)

Read operation (CL=ONE):
- Read from Node A or B → Success
- Might get stale data if reading from Node C (when it recovers)
- ❌ Not strongly consistent

Result: AP (Available but eventually consistent)
\`\`\`

**Scenario 2: Consistency Level QUORUM (CP Behavior)**

\`\`\`
Nodes: A, B, C (replication factor = 3)
Need 2/3 nodes for QUORUM

Write operation (CL=QUORUM):
- Write to Nodes A, B → Success
- Must wait for 2 acknowledgments
- Node C down → Doesn't matter (have 2/3)
- ✅ Write succeeds

Now Node B also fails:

Write operation (CL=QUORUM):
- Only Node A available
- Need 2 nodes, only have 1
- ❌ Write fails (unavailable)
- ✅ But data remains consistent

Result: CP (Consistent but unavailable when quorum lost)
\`\`\`

**Scenario 3: Consistency Level ALL (Strictly CP)**

\`\`\`
Nodes: A, B, C
Node C fails

Write operation (CL=ALL):
- Need all 3 nodes
- Only have A and B
- ❌ Write fails

Result: Highly consistent but low availability
\`\`\`

**Mathematical Explanation:**

For strong consistency with quorum:

\`\`\`
W + R > N

Where:
- W = Write quorum
- R = Read quorum  
- N = Replication factor

Example: W=2, R=2, N=3
- 2 + 2 > 3 ✓ (provides strong consistency)
- If 2 nodes respond to write, and you read from 2 nodes,
  at least 1 node is guaranteed to have the latest write

For eventual consistency:
- W=1, R=1, N=3
- 1 + 1 = 2 (NOT > 3)
- No guarantee latest write is seen
\`\`\`

**Why This Matters:**

**1. Cassandra defaults to AP (ONE/LOCAL_ONE)**
\`\`\`
// Default behavior
session.execute (query); // Uses consistency level ONE
→ High availability, eventual consistency
→ AP classification
\`\`\`

**2. But can be configured for CP (QUORUM/ALL)**
\`\`\`
// Strong consistency
session.execute (query, ConsistencyLevel.QUORUM);
→ Requires majority, unavailable if quorum lost
→ CP classification
\`\`\`

**Real-World Example:**

**Instagram\'s Cassandra Usage:**

\`\`\`
User Posts (AP - Consistency Level ONE):
- High write throughput needed
- Slight delay in seeing posts acceptable
- Better to accept post than show error

User Blocks (CP - Consistency Level QUORUM):
- If Alice blocks Bob, Bob must not see Alice's posts immediately
- Consistency critical for privacy/safety
- Use QUORUM to ensure block is replicated
\`\`\`

**Your Scenario Explained:**

"When you use consistency level QUORUM, the system sometimes becomes unavailable when nodes are down."

This is **expected behavior** because QUORUM requires majority of nodes:

\`\`\`
Replication Factor: 3
QUORUM needs: 2 nodes

1 node down: ✅ Still have 2/3 (QUORUM available)
2 nodes down: ❌ Only 1/3 (QUORUM unavailable)
\`\`\`

**When you use QUORUM, you're choosing CP behavior:**
- ✅ Consistent: Guaranteed to read latest write
- ❌ Unavailable: If majority nodes down

**Is Cassandra AP or CP?**

**Technically correct answer:** "It depends on consistency level."

**Practical answer:**
- **Designed as AP**: Default behavior is AP (CL=ONE, high availability)
- **Can be configured as CP**: Using QUORUM/ALL consistency levels

**In Interviews, Say:**

"Cassandra is typically classified as AP because its default configuration prioritizes availability with eventual consistency. However, it supports tunable consistency levels. When you use QUORUM for reads and writes, you get CP behavior - strong consistency but potential unavailability if you lose quorum. This flexibility is powerful: you can use CL=ONE for high-throughput, non-critical writes (AP) and CL=QUORUM for critical data requiring consistency (CP)."

**Trade-offs of Different Consistency Levels:**

\`\`\`
ONE (AP):
✅ Highest availability
✅ Lowest latency
✅ Best write throughput
❌ Stale reads possible
❌ Eventually consistent

QUORUM (CP):
✅ Strong consistency (W + R > N)
✅ Balanced
⚠️ Medium latency
⚠️ Unavailable if quorum lost
❌ Lower throughput

ALL (Strictly CP):
✅ Strongest consistency
❌ Highest latency
❌ Lowest availability (any node failure = unavailable)
❌ Lowest throughput
\`\`\`

**Best Practice:**

Use **different consistency levels for different operations**:

\`\`\`
// High-volume, non-critical
INSERT INTO user_events (CL=ONE);  // AP behavior

// Critical data
INSERT INTO user_balance (CL=QUORUM);  // CP behavior

// Critical read
SELECT balance WHERE user_id = X (CL=QUORUM);  // CP behavior

// Non-critical read
SELECT posts WHERE user_id = X (CL=ONE);  // AP behavior
\`\`\`

**Key Insight:**

The nuance is that **CAP is not a database property but a choice you make per operation**. Cassandra gives you the flexibility to choose. Most systems are "AP" because that's the default, but sophisticated users tune consistency per query based on requirements.`,
    keyPoints: [
      'Cassandra is tunable - not strictly AP or CP, depends on consistency level',
      'Consistency Level ONE → AP (high availability, eventual consistency)',
      'Consistency Level QUORUM → CP (strong consistency, unavailable without majority)',
      'Strong consistency requires W + R > N (write quorum + read quorum > replication factor)',
      'Cassandra defaults to AP but can be configured for CP per query',
      'Use different consistency levels for different operations based on requirements',
      'Flexibility allows: ONE for non-critical (AP), QUORUM for critical (CP)',
    ],
  },
];
