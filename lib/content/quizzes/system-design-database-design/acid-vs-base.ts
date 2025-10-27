/**
 * Quiz questions for ACID vs BASE Properties section
 */

export const acidvsbaseQuiz = [
  {
    id: 'acid-base-disc-q1',
    question:
      'Your startup is building a ride-sharing app. The payments team insists on ACID for all data to ensure reliability. The engineering team wants to use Cassandra (BASE) for everything to handle scale. How would you resolve this conflict? Discuss which data needs ACID and which can use BASE.',
    sampleAnswer: `I would propose a **hybrid architecture** that uses ACID where critical and BASE where appropriate. The key is understanding that different data in the same application has different consistency requirements.

**Requirements Analysis:**

**Ride-sharing app data types:**1. Payments (ride charges)
2. Driver locations (real-time tracking)
3. User profiles
4. Ride history
5. Trip records (current rides)
6. Driver ratings/reviews

**ACID Data (PostgreSQL):**

**1. Payments & Billing**
\`\`\`
Why ACID required:
- Must never double-charge user
- Must never lose payment record
- Charge + receipt must be atomic
- Balance deductions must be consistent

Example transaction:
BEGIN TRANSACTION;
  -- Charge user
  INSERT INTO charges (user_id, amount, status) VALUES (123, 25.00, 'pending');
  
  -- Update driver earnings
  UPDATE driver_accounts SET balance = balance + 22.50 WHERE driver_id = 456;
  
  -- Create receipt
  INSERT INTO receipts (charge_id, user_id, amount) VALUES (LAST_INSERT_ID(), 123, 25.00);
  
  -- If any step fails, ROLLBACK all
COMMIT;

ACID properties needed:
- Atomicity: All steps succeed or all fail
- Consistency: Balance constraints enforced
- Isolation: Concurrent charges don't interfere
- Durability: Payment recorded even if crash
\`\`\`

**2. Active Ride Records**
\`\`\`
Why ACID required:
- One driver cannot have two active rides
- Ride assignment must be atomic
- Prevent double-booking

Example transaction:
BEGIN TRANSACTION;
  -- Check driver availability
  SELECT status FROM drivers WHERE id = 456 FOR UPDATE;
  -- status = 'available'
  
  -- Assign ride
  UPDATE drivers SET status = 'on_ride', current_ride_id = 789 WHERE id = 456;
  
  -- Create ride record
  INSERT INTO rides (id, driver_id, rider_id, status) VALUES (789, 456, 123, 'in_progress');
  
COMMIT;

ACID properties needed:
- Atomicity: Assignment + ride creation atomic
- Isolation: Prevents double-booking same driver
\`\`\`

**BASE Data (Cassandra/DynamoDB):**

**1. Driver Location Tracking**
\`\`\`
Why BASE acceptable:
- Updates every 2-3 seconds
- Location being 1-2 seconds stale is acceptable
- High write throughput (millions of updates/sec)
- Availability critical (must always show drivers)

Cassandra schema:
CREATE TABLE driver_locations (
  driver_id UUID,
  timestamp TIMESTAMP,
  lat DOUBLE,
  lon DOUBLE,
  PRIMARY KEY (driver_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

Write (CL=ONE):
- Low latency (5ms)
- Async replication
- Eventually consistent

Read (CL=ONE):
- Nearest replica
- Might be 1-2 seconds stale
- Acceptable for map display
\`\`\`

**2. Ride History**
\`\`\`
Why BASE acceptable:
- Historical data (read-heavy)
- Slight delay in showing completed rides acceptable
- High read throughput required
- Availability important (users checking history)

Storage: Cassandra
Partition key: user_id
Clustering key: ride_timestamp

Eventual consistency acceptable:
- Ride appears in history 1-2 seconds after completion
- User doesn't expect instant historical updates
\`\`\`

**3. Driver Ratings & Reviews**
\`\`\`
Why BASE acceptable:
- Not time-critical
- Average rating being off by 0.01 is acceptable
- High read volume
- Availability important

Storage: DynamoDB
Consistency: Eventually consistent reads (default)

Trade-off:
- Fast reads (1-5ms)
- Slightly stale average ratings
- Fine for this use case
\`\`\`

**4. User Profiles**
\`\`\`
Why BASE acceptable:
- Updates infrequent
- Read-heavy
- User sees own updates (read-your-writes consistency)
- Other users seeing stale profile for 1 second is fine

Storage: DynamoDB with session consistency
- User who updates profile sees changes immediately
- Other users see changes after 1-2 seconds (eventual)
\`\`\`

**Architecture Summary:**

\`\`\`
PostgreSQL (ACID):
├─ Payments & charges
├─ Active rides (in-progress)
├─ Driver availability status
└─ Financial records

Cassandra (BASE):
├─ Driver locations (time-series)
├─ Ride history (completed rides)
└─ Activity logs

DynamoDB (BASE):
├─ User profiles
├─ Driver profiles
└─ Ratings & reviews

Redis (Cache):
├─ Session storage
└─ Driver location cache (hot data)
\`\`\`

**Addressing Team Concerns:**

**To Payments Team:**
"You're right that payments require ACID. We'll use PostgreSQL for all financial transactions, active rides, and driver availability. This ensures atomicity (charges + receipts succeed together), consistency (balances can't go negative), and durability (no lost payments). This is non-negotiable for financial data."

**To Engineering Team:**
"You're right that scale is important. We'll use Cassandra/DynamoDB for high-volume, real-time data like driver locations (millions of updates/sec) and ride history (billions of records). This gives us the scalability we need without compromising financial accuracy. It\'s not all-or-nothing; we use the right tool for each job."

**Trade-Offs Accepted:**

**PostgreSQL (ACID):**
- ✅ Financial accuracy
- ✅ No double-booking
- ✅ Data integrity
- ❌ Harder to scale horizontally
- ❌ Higher latency for writes (~10ms vs 1ms)

**Cassandra/DynamoDB (BASE):**
- ✅ Handles millions of writes/sec
- ✅ Scales to billions of records
- ✅ Low latency (1-5ms)
- ❌ Eventual consistency (1-2 second staleness)
- ❌ More complex application logic

**Cost Comparison:**

**All ACID (PostgreSQL):**
- Driver locations: Can't handle write throughput
- Would need massive vertical scaling (\$10K+/month)
- Still might not meet latency requirements

**All BASE (Cassandra):**
- Payments: Risk of double-charging users
- Active rides: Risk of double-booking drivers
- Financial/legal liability
- Trust issues

**Hybrid (Recommended):**
- PostgreSQL: ~$500/month (modest scale for financial data)
- Cassandra: ~$1,000/month (handles millions of location updates)
- Total: $1,500/month
- Best of both worlds

**Real-World Validation:**

**Uber actually uses this hybrid approach:**
- PostgreSQL: Payments, active trips, driver state
- Cassandra: Driver locations, ride history, analytics
- Redis: Caching, session storage

**Key Insight:**

The conflict is based on a false dichotomy. You don't choose ACID or BASE for the entire system - you choose based on requirements for each type of data. Financial data requires ACID (correctness over everything), while location tracking can use BASE (availability and scale matter more than perfect consistency). The hybrid approach satisfies both teams' legitimate concerns.`,
    keyPoints: [
      'Use hybrid architecture: ACID for critical data, BASE for high-volume data',
      'Payments require ACID (PostgreSQL) - atomicity and consistency non-negotiable',
      'Driver locations can use BASE (Cassandra) - eventual consistency acceptable',
      'Active rides require ACID - prevent double-booking with isolation',
      "Ride history can use BASE - historical data doesn't need strong consistency",
      'Hybrid approach provides best of both worlds: correctness + scalability',
      'Uber uses this exact pattern in production',
    ],
  },
  {
    id: 'acid-base-disc-q2',
    question:
      'Your database team runs all transactions at SERIALIZABLE isolation level "to be safe." This causes performance issues and deadlocks. When is SERIALIZABLE actually necessary vs when can you use READ COMMITTED? Provide specific examples.',
    sampleAnswer: `Running everything at SERIALIZABLE is overkill and causes unnecessary performance issues. Let me explain when each isolation level is appropriate.

**The Problem with Always Using SERIALIZABLE:**

\`\`\`
Performance impact:
- 10-100x slower than READ COMMITTED
- Increased lock contention
- More deadlocks (transactions abort and retry)
- Lower throughput
- Resource intensive (more locks, more waiting)

Reality: Most operations don't need full serializability
\`\`\`

**When SERIALIZABLE is Actually Necessary:**

**1. Preventing Double-Booking**

\`\`\`sql
-- Concert ticket sales: Only 100 seats available

-- With READ COMMITTED (WRONG):
BEGIN TRANSACTION;
  SELECT COUNT(*) FROM bookings WHERE concert_id = 1;
  -- Returns 99
  
  -- Another transaction also reads 99 (not isolated)
  -- Both insert, result: 101 bookings (overbooked!)
  
  INSERT INTO bookings (concert_id, user_id, seat) VALUES (1, 123, 'A1');
COMMIT;

-- With SERIALIZABLE (CORRECT):
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
  SELECT COUNT(*) FROM bookings WHERE concert_id = 1 FOR UPDATE;
  -- Returns 99, locks table
  
  -- Other transactions wait
  
  IF count < 100 THEN
    INSERT INTO bookings (concert_id, user_id, seat) VALUES (1, 123, 'A1');
  END IF;
COMMIT;

Why SERIALIZABLE needed:
- Phantom reads possible with READ COMMITTED
- Two transactions both see 99, both insert
- SERIALIZABLE prevents this with range locks
\`\`\`

**2. Bank Account Transfers with Constraints**

\`\`\`sql
-- Business rule: Combined balance of checking + savings >= $100

-- With READ COMMITTED (WRONG):
Transaction A: Transfer $500 from checking to external account
  SELECT checking_balance FROM accounts WHERE user_id = 1;  -- $600
  -- Checking: $600, Savings: $50, Total: $650 ✓
  
Transaction B (concurrent): Transfer $500 from savings to external
  SELECT savings_balance FROM accounts WHERE user_id = 1;  -- $50
  -- But doesn't see A's pending transfer
  
Both commit, result:
  Checking: $100
  Savings: -$450 (NEGATIVE!)
  Total: -$350 (violates constraint)

-- With SERIALIZABLE (CORRECT):
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
  SELECT checking_balance, savings_balance FROM accounts WHERE user_id = 1 FOR UPDATE;
  -- Locks both accounts
  
  IF (checking - 500) + savings >= 100 THEN
    UPDATE accounts SET checking = checking - 500 WHERE user_id = 1;
  ELSE
    ROLLBACK;
  END IF;
COMMIT;

Why SERIALIZABLE needed:
- Must evaluate constraint across multiple rows atomically
- READ COMMITTED allows interleaving that violates constraint
\`\`\`

**3. Inventory with Multiple Warehouses**

\`\`\`sql
-- Product must have total quantity >= 10 across all warehouses

-- With READ COMMITTED (WRONG):
SELECT SUM(quantity) FROM inventory WHERE product_id = 'P123';
-- Returns 15 (warehouse A:10, B:5)

UPDATE inventory SET quantity = quantity - 8 WHERE warehouse = 'A';
-- During this update, another transaction reduces B by 3
-- Final: A:2, B:2, Total:4 (violates constraint!)

-- With SERIALIZABLE (CORRECT):
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
  SELECT SUM(quantity) FROM inventory WHERE product_id = 'P123' FOR UPDATE;
  -- Locks all rows for this product
  
  IF total >= 10 THEN
    UPDATE inventory SET quantity = quantity - 8 WHERE warehouse = 'A';
  END IF;
COMMIT;

Why SERIALIZABLE needed:
- Aggregate constraints across multiple rows
- Must prevent other transactions from modifying any warehouse quantity
\`\`\`

**When READ COMMITTED is Sufficient:**

**1. Simple Balance Check (Single Account)**

\`\`\`sql
-- Transfer money from one account to another

BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
  SELECT balance FROM accounts WHERE id = 'A' FOR UPDATE;
  -- Locks single row
  
  IF balance >= 100 THEN
    UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
    UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
  ELSE
    ROLLBACK;
  END IF;
COMMIT;

Why READ COMMITTED sufficient:
- Constraint on single row (account A balance)
- FOR UPDATE locks that row
- No phantom reads possible
- Much faster than SERIALIZABLE
\`\`\`

**2. User Profile Updates**

\`\`\`sql
-- User updates their profile

BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
  UPDATE users SET 
    email = 'newemail@example.com',
    name = 'New Name',
    updated_at = NOW()
  WHERE id = 123;
COMMIT;

Why READ COMMITTED sufficient:
- Single row update
- No complex constraints
- No aggregates or ranges
- No other transactions care about this user's profile
\`\`\`

**3. Order Creation (with Foreign Keys)**

\`\`\`sql
-- Create order for existing user

BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
  -- Foreign key ensures user exists
  INSERT INTO orders (user_id, total, status) 
  VALUES (123, 99.99, 'pending');
  
  INSERT INTO order_items (order_id, product_id, quantity, price)
  VALUES (LAST_INSERT_ID(), 'P456', 2, 49.99);
COMMIT;

Why READ COMMITTED sufficient:
- Foreign key constraint handles validation
- New rows, no conflicts with other transactions
- Atomicity provided by transaction
- SERIALIZABLE would be overkill
\`\`\`

**4. Analytics Queries (Read-Only)**

\`\`\`sql
-- Generate daily sales report

BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
  SELECT 
    DATE(created_at) as date,
    SUM(total) as daily_revenue,
    COUNT(*) as order_count
  FROM orders
  WHERE created_at >= '2024-01-01'
  GROUP BY DATE(created_at);
COMMIT;

Why READ COMMITTED sufficient:
- Read-only query
- Slight inconsistency acceptable for analytics
- REPEATABLE READ or SERIALIZABLE would lock many rows
- Not worth performance cost
\`\`\`

**5. Logging/Audit Trail**

\`\`\`sql
-- Log user action

BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
  INSERT INTO audit_log (user_id, action, timestamp, details)
  VALUES (123, 'login', NOW(), '{"ip": "1.2.3.4"}');
COMMIT;

Why READ COMMITTED sufficient:
- Append-only operation
- No conflicts possible
- Order doesn't matter
- SERIALIZABLE unnecessary overhead
\`\`\`

**Decision Matrix:**

| Scenario | Isolation Level | Why |
|----------|----------------|-----|
| **Double-booking prevention** | SERIALIZABLE | Phantom reads must be prevented |
| **Multi-row constraints** | SERIALIZABLE | Must lock range, evaluate atomically |
| **Aggregate constraints** | SERIALIZABLE | Must prevent changes to any involved rows |
| **Single-row constraint** | READ COMMITTED + FOR UPDATE | Sufficient, much faster |
| **Profile updates** | READ COMMITTED | No conflicts, single row |
| **Order creation** | READ COMMITTED | Foreign keys handle validation |
| **Analytics (read-only)** | READ COMMITTED | Slight inconsistency acceptable |
| **Logging** | READ COMMITTED | Append-only, no conflicts |

**Recommended Strategy:**

\`\`\`typescript
// Default: READ COMMITTED (90% of transactions)
const result = await db.query(\`
  UPDATE users SET email = $1 WHERE id = $2
\`, [email, userId]);

// SERIALIZABLE: Only when necessary (10% of transactions)
await db.query('BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE');
try {
  const seats = await db.query(\`
    SELECT COUNT(*) FROM bookings WHERE concert_id = $1 FOR UPDATE
  \`, [concertId]);
  
  if (seats[0].count < MAX_CAPACITY) {
    await db.query(\`
      INSERT INTO bookings (concert_id, user_id) VALUES ($1, $2)
    \`, [concertId, userId]);
    await db.query('COMMIT');
  } else {
    await db.query('ROLLBACK');
  }
} catch (err) {
  await db.query('ROLLBACK');
  throw err;
}
\`\`\`

**Performance Comparison:**

\`\`\`
Simple user profile update:
- READ COMMITTED: 5ms
- SERIALIZABLE: 50ms (10x slower)

Ticket booking (high contention):
- READ COMMITTED: Overbooking bugs
- SERIALIZABLE: 100ms but correct

Recommended:
- Use READ COMMITTED as default
- Upgrade to SERIALIZABLE only for critical operations
- Result: 90% of operations fast, 10% correct but slower
\`\`\`

**Key Takeaway for Database Team:**

"SERIALIZABLE is necessary for about 10% of our transactions - those involving:
1. Range queries with constraints (double-booking prevention)
2. Multi-row constraints (total balance across accounts)
3. Aggregate-based decisions (inventory across warehouses)

For the other 90% (single-row updates, profile changes, logging), READ COMMITTED with FOR UPDATE is sufficient and 10x faster. We should default to READ COMMITTED and explicitly use SERIALIZABLE only where correctness requires it. This will eliminate most deadlocks and dramatically improve performance while maintaining correctness where it matters."`,
    keyPoints: [
      'SERIALIZABLE necessary for: double-booking, multi-row constraints, aggregate-based decisions',
      'READ COMMITTED sufficient for: single-row updates, profile changes, logging, most operations',
      'SERIALIZABLE is 10-100x slower than READ COMMITTED',
      'Default to READ COMMITTED, upgrade to SERIALIZABLE only when needed',
      'Use FOR UPDATE with READ COMMITTED for single-row constraints',
      'Performance: Use SERIALIZABLE for ~10% of transactions, READ COMMITTED for 90%',
      'Always using SERIALIZABLE causes unnecessary deadlocks and performance issues',
    ],
  },
  {
    id: 'acid-base-disc-q3',
    question:
      "Amazon\'s shopping cart famously uses eventual consistency (BASE). Sometimes users see different cart contents on different devices until sync completes. Why is this acceptable for carts but not for checkout/payment? Explain the trade-offs.",
    sampleAnswer: `Great question! Amazon's choice to use eventual consistency for shopping carts but strong consistency for checkout demonstrates perfect understanding of when each model is appropriate. Let me explain the trade-offs.

**Why Eventual Consistency is OK for Shopping Carts:**

**Requirements Analysis:**

\`\`\`
Shopping cart characteristics:
- High write frequency (users constantly adding/removing items)
- Multi-device access (phone, tablet, desktop)
- Temporary data (abandoned carts common)
- Availability critical (cart must always work)
- Perfect consistency not required (slight staleness acceptable)
\`\`\`

**Architecture (BASE / Eventual Consistency):**

\`\`\`
User adds item on phone:
1. Write to DynamoDB (nearest region)
2. Acknowledge immediately to user (fast)
3. Replicate to other regions asynchronously
4. User on tablet sees change after 1-2 seconds

Trade-off accepted:
- Brief inconsistency (different devices see different cart briefly)
- High availability (cart always works, even during partitions)
- Fast response time (<10ms)
\`\`\`

**Why This is Acceptable:**

**1. No Financial Impact**

\`\`\`
Cart is pre-purchase:
- No money charged
- No inventory committed
- Just a "wish list" until checkout
- User can review cart at checkout
- If items disappeared (merged from different devices), user notices before paying
\`\`\`

**2. User Understands Context**

\`\`\`
Users implicitly understand:
- "I added this on my phone, might not be on tablet yet"
- Similar to physical shopping: Items in cart on aisle 3, go to aisle 7, cart still has them
- When you return to aisle 3, cart synchronized

Users are forgiving of brief inconsistency in pre-purchase state
\`\`\`

**3. Conflict Resolution is Straightforward**

\`\`\`
Conflict scenario:
- User on phone: Adds Item A
- User on tablet: Adds Item B
- Both happen concurrently

Resolution:
- Merge carts: {Item A, Item B}
- Use LWW (Last-Write-Wins) for item quantities
- Show merged cart at checkout
- User can adjust if needed

Result: No data loss, simple merge logic
\`\`\`

**4. High Availability Requirement**

\`\`\`
Business requirement:
- Cart must always work
- Cannot show "Cart unavailable" during network issues
- Lost sales if users can't add items

Eventual consistency provides:
- Always available (works during partitions)
- Fast response times (no coordination needed)
- Scales to millions of users
\`\`\`

**Why Strong Consistency Required for Checkout:**

**1. Financial Transaction**

\`\`\`
Checkout involves:
- Charging payment method
- Committing inventory
- Creating legal order contract
- Financial and legal liability

Requirements:
- Must charge exact amount shown
- Cannot charge twice
- Cannot commit inventory we don't have
- Must create order record atomically

This requires ACID transactions (strong consistency)
\`\`\`

**2. Inventory Validation**

\`\`\`
At checkout, must verify:
- Items still in stock
- Prices haven't changed
- Product still available

With eventual consistency:
User sees 5 items in cart → tries to checkout
Item 1: Out of stock (inventory sold out during replication lag)
Item 2: Price changed $50 → $60 (update not yet propagated)

This is unacceptable at payment stage
\`\`\`

**3. Atomicity Required**

\`\`\`
Checkout transaction:
BEGIN TRANSACTION;
  -- 1. Validate inventory
  SELECT quantity FROM products WHERE id = 'P123' FOR UPDATE;
  
  -- 2. Deduct inventory
  UPDATE products SET quantity = quantity - 1 WHERE id = 'P123';
  
  -- 3. Charge payment
  INSERT INTO charges (user_id, amount) VALUES (123, 59.99);
  
  -- 4. Create order
  INSERT INTO orders (user_id, total, status) VALUES (123, 59.99, 'paid');
  
  -- All steps must succeed together or rollback
COMMIT;

If payment fails: Rollback inventory deduction
If inventory unavailable: Don't charge payment

ACID atomicity ensures consistency
\`\`\`

**4. Legal and Trust Implications**

\`\`\`
Checkout problems with eventual consistency:
- User charged $100, receipt shows $90 (inconsistent)
  → Legal issue, fraud complaint
  
- Item shown as purchased, but inventory not deducted
  → Overselling, cannot fulfill order
  → Customer dissatisfaction
  
- Payment charged twice (duplicate during replication)
  → Refund required, trust lost

Trust issues cost more than strong consistency overhead
\`\`\`

**The Architecture Difference:**

**Shopping Cart (Eventual Consistency / BASE):**

\`\`\`
Storage: DynamoDB (multi-region replication)
Consistency: Eventually consistent reads
Write: Async replication

Add to cart:
- Write to nearest region (5ms)
- Acknowledge immediately
- Replicate asynchronously
- User sees update on same device immediately (session consistency)
- Other devices see update after 1-2 seconds

Conflict resolution:
- LWW (Last-Write-Wins) for item quantities
- Union merge for different items
- Show merged result at checkout

Benefits:
- <10ms latency
- Always available
- Handles network partitions
- Scales globally

Trade-offs:
- Cart may differ across devices for 1-2 seconds
- Acceptable because pre-purchase
\`\`\`

**Checkout (Strong Consistency / ACID):**

\`\`\`
Storage: PostgreSQL (or Aurora with strong consistency)
Consistency: ACID transactions
Isolation: READ COMMITTED or REPEATABLE READ

Checkout flow:
BEGIN TRANSACTION;
  -- Lock inventory
  SELECT quantity FROM products WHERE id = $1 FOR UPDATE;
  
  IF quantity >= requested_quantity THEN
    -- Deduct inventory atomically
    UPDATE products SET quantity = quantity - $2 WHERE id = $1;
    
    -- Charge payment
    INSERT INTO charges (user_id, amount, status) VALUES ($3, $4, 'charged');
    
    -- Create order
    INSERT INTO orders (user_id, total, status) VALUES ($3, $4, 'paid');
    
    COMMIT;
  ELSE
    ROLLBACK; -- Insufficient inventory
    RETURN 'Out of stock';
  END IF;

Benefits:
- Atomic (all-or-nothing)
- Consistent (constraints enforced)
- Isolated (no double-selling)
- Durable (order recorded)

Trade-offs:
- 50-100ms latency (slower than cart)
- Less available during failures
- Acceptable because correctness critical
\`\`\`

**Cart to Checkout Transition:**

\`\`\`
User flow:
1. Browse products → Add to cart (eventual consistency, fast)
2. Continue shopping → Modify cart (eventual consistency, fast)
3. Click "Checkout" → Validate cart against current prices/inventory (strong consistency)
4. Review order → Show accurate totals (strong consistency)
5. Submit payment → Execute transaction (strong consistency, ACID)

Transition point: "Checkout" button
- Before checkout: Eventual consistency OK
- After checkout: Strong consistency required

Validation at checkout:
- Fetch current prices from PostgreSQL (strongly consistent)
- Verify inventory available
- Show warnings if cart items changed:
  "Price changed: $50 → $45" ← User can accept or cancel
  "Item no longer available" ← Remove from cart
\`\`\`

**Trade-off Summary:**

| Aspect | Shopping Cart | Checkout |
|--------|--------------|----------|
| **Consistency** | Eventual (1-2 sec lag) | Strong (immediate) |
| **Database** | DynamoDB | PostgreSQL |
| **Latency** | <10ms | 50-100ms |
| **Availability** | Always available | May be unavailable during failures |
| **Atomicity** | Not required | Required (ACID) |
| **Business Impact if Wrong** | Minor (user sees cart sync) | Major (double-charge, overselling) |

**Why This Hybrid Approach Works:**

\`\`\`
User spends 95% of time shopping (add/remove from cart):
- Fast, responsive cart (10ms)
- Always available
- Eventual consistency acceptable

User spends 5% of time checking out:
- 100ms checkout is acceptable (one-time)
- Strong consistency required (financial)
- Lower availability acceptable (retry if fails)

Result: Optimize for common case (shopping) while ensuring correctness for critical case (payment)
\`\`\`

**Real Numbers (Amazon Scale):**

\`\`\`
Shopping cart operations: 100M+ per day
- Eventual consistency: <10ms latency
- High availability: 99.99%
- Cost: Low (DynamoDB scales easily)

Checkout operations: 1M per day (1% conversion)
- Strong consistency: 50-100ms latency
- Lower availability: 99.9% (acceptable for 1%)
- Cost: Higher (PostgreSQL, more expensive)

If reversed:
- Cart with strong consistency: 10x slower, costs 10x more, no benefit
- Checkout with eventual consistency: Financial disasters
\`\`\`

**Key Insight:**

Shopping cart consistency requirements are fundamentally different from checkout:
- **Cart**: Pre-purchase, temporary, recoverable, high-volume → Eventual consistency optimal
- **Checkout**: Financial, legal, permanent, critical → Strong consistency required

The trade-off is accepting brief cart inconsistency (1-2 seconds across devices) for massive performance and availability gains, while ensuring perfect consistency where money and inventory are involved. This demonstrates mature system design: understanding that different data in the same application requires different consistency guarantees.`,
    keyPoints: [
      'Shopping cart uses eventual consistency (BASE) - acceptable because pre-purchase, no financial impact',
      'Checkout requires strong consistency (ACID) - financial transaction, inventory commitment',
      'Cart inconsistency (1-2 seconds) acceptable - users tolerate slight staleness before purchase',
      'Checkout atomicity critical - charge + inventory + order must succeed together',
      'Hybrid approach optimizes for common case (shopping) while ensuring correctness (payment)',
      'Cart: DynamoDB eventual consistency (<10ms, always available)',
      'Checkout: PostgreSQL ACID (50-100ms, strongly consistent)',
      'Validate cart at checkout transition - show price/availability changes',
    ],
  },
];
