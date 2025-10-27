/**
 * ACID vs BASE Properties Section
 */

export const acidvsbaseSection = {
  id: 'acid-vs-base',
  title: 'ACID vs BASE Properties',
  content: `ACID and BASE represent two contrasting philosophies for database transaction management. ACID prioritizes consistency and reliability, while BASE prioritizes availability and scalability. Understanding both is crucial for choosing the right database and consistency model.

## ACID Properties

**ACID** stands for: **Atomicity, Consistency, Isolation, Durability**

These properties guarantee reliable transaction processing in traditional relational databases.

### **A - Atomicity**

**Definition**: A transaction is treated as a single, indivisible unit. Either all operations succeed (commit) or all fail (rollback).

**The "All or Nothing" Principle**

\`\`\`sql
BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 'Alice';
  UPDATE accounts SET balance = balance + 100 WHERE id = 'Bob';
COMMIT;

If either UPDATE fails (e.g., insufficient balance):
- ROLLBACK both operations
- Database returns to state before transaction
- No partial state (Alice debited but Bob not credited)
\`\`\`

**Real-World Example:**

\`\`\`
E-commerce order placement:
1. Deduct item from inventory
2. Create order record
3. Charge payment
4. Send confirmation email

If step 3 (payment) fails:
- Rollback steps 1 and 2
- Inventory restored
- No order record created
- User not charged
\`\`\`

**Why Atomicity Matters:**
- Prevents partial updates that leave system in inconsistent state
- Critical for financial transactions
- Simplifies error handling (don't need to manually undo operations)

---

### **C - Consistency**

**Definition**: A transaction moves the database from one valid state to another, maintaining all defined rules (constraints, triggers, cascades).

**Database Constraints Enforced:**
- **Primary keys**: No duplicate IDs
- **Foreign keys**: References must exist
- **Check constraints**: Values must meet conditions
- **Unique constraints**: No duplicate values in column
- **Not null**: Required fields must have values

\`\`\`sql
CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT NOT NULL,
  total DECIMAL CHECK (total >= 0),  -- Must be non-negative
  FOREIGN KEY (user_id) REFERENCES users (id)
);

-- This transaction will FAIL and ROLLBACK:
INSERT INTO orders (id, user_id, total)
VALUES (1, 999, -50);
-- Reason 1: user_id 999 doesn't exist (foreign key violation)
-- Reason 2: total is negative (check constraint violation)

Database remains consistent - no invalid order created
\`\`\`

**Real-World Example:**

\`\`\`
Bank account transfer:
- Business rule: Balance cannot go negative
- Account A has $50
- Transaction: Transfer $100 from A to B

BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
  -- This violates consistency rule (balance would be -$50)
ROLLBACK;

Result: Transaction rejected, database remains consistent
\`\`\`

**Why Consistency Matters:**
- Ensures data integrity
- Prevents invalid states
- Enforces business rules at database level

---

### **I - Isolation**

**Definition**: Concurrent transactions execute independently without interfering with each other. Each transaction appears to execute in isolation.

**The Problem Without Isolation:**

\`\`\`
Transaction 1: Read balance ($100) → Add $50 → Write $150
Transaction 2: Read balance ($100) → Add $30 → Write $130

Without isolation:
- T1 reads $100
- T2 reads $100 (dirty read - T1 hasn't committed)
- T1 writes $150
- T2 writes $130 (overwrites T1's change)
Result: Lost update! Should be $180, but is $130
\`\`\`

**Isolation Levels** (from weakest to strongest):

#### **1. Read Uncommitted** (Weakest Isolation)

**Allows**: Dirty reads (reading uncommitted changes)

\`\`\`
Transaction A: UPDATE accounts SET balance = 1000 WHERE id = 'Alice';
Transaction B: SELECT balance FROM accounts WHERE id = 'Alice';
-- Returns 1000 even though A hasn't committed yet

Transaction A: ROLLBACK;
-- B read data that was never committed (dirty read)
\`\`\`

**Problems**: Dirty reads
**Use case**: Rarely used (only for approximate analytics where accuracy not critical)

#### **2. Read Committed** (Default in PostgreSQL, Oracle)

**Prevents**: Dirty reads (only see committed data)
**Allows**: Non-repeatable reads

\`\`\`
Transaction A:
  SELECT balance FROM accounts WHERE id = 'Alice';  -- Returns $100

  -- Meanwhile, Transaction B commits:
  UPDATE accounts SET balance = 200 WHERE id = 'Alice';
  COMMIT;

  SELECT balance FROM accounts WHERE id = 'Alice';  -- Returns $200
  -- Same query, different result within same transaction!
\`\`\`

**Problems**: Non-repeatable reads (same query returns different results)
**Use case**: Most common default, good balance

#### **3. Repeatable Read** (Default in MySQL)

**Prevents**: Dirty reads, non-repeatable reads
**Allows**: Phantom reads

\`\`\`
Transaction A:
  SELECT COUNT(*) FROM orders WHERE amount > 100;  -- Returns 5

  -- Meanwhile, Transaction B commits new order:
  INSERT INTO orders VALUES (..., amount = 150);
  COMMIT;

  SELECT COUNT(*) FROM orders WHERE amount > 100;  -- Returns 6
  -- New rows appeared (phantom read)
\`\`\`

**Problems**: Phantom reads (new rows appear)
**Use case**: When you need consistent reads within transaction

#### **4. Serializable** (Strongest Isolation)

**Prevents**: All concurrency anomalies
**Effect**: Transactions execute as if serial (one after another)

\`\`\`
Transaction A and B both try to update same row:
- A acquires lock
- B waits for A to complete
- After A commits, B proceeds
Result: No conflicts, full isolation
\`\`\`

**Cost**: Performance penalty, potential deadlocks
**Use case**: Financial transactions, when correctness is paramount

**Isolation Level Comparison:**

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|-------|-----------|-------------------|-------------|-------------|
| **Read Uncommitted** | ✗ Possible | ✗ Possible | ✗ Possible | Fastest |
| **Read Committed** | ✓ Prevented | ✗ Possible | ✗ Possible | Fast |
| **Repeatable Read** | ✓ Prevented | ✓ Prevented | ✗ Possible | Slower |
| **Serializable** | ✓ Prevented | ✓ Prevented | ✓ Prevented | Slowest |

---

### **D - Durability**

**Definition**: Once a transaction is committed, it remains committed even in case of system failure (crash, power loss).

**How Durability is Achieved:**

#### **Write-Ahead Logging (WAL)**

\`\`\`
Transaction commits:
1. Write changes to WAL (sequential write, fast)
2. Flush WAL to disk (fsync)
3. Return "committed" to client
4. Later: Apply changes to data files (background process)

System crashes before step 4:
- On restart, replay WAL
- Recover all committed transactions
- Database restored to consistent state
\`\`\`

**Real-World Example:**

\`\`\`
User completes payment transaction
System responds: "Payment successful"
Power outage occurs immediately after

With Durability:
- Transaction logged in WAL
- After restart, transaction recovered
- Payment recorded, user's order safe

Without Durability:
- Transaction lost
- User charged but no order created
- Data inconsistency
\`\`\`

**Durability Techniques:**
- **Write-Ahead Logging (WAL)**: Log changes before applying
- **Replication**: Synchronous replication to replica
- **Fsync**: Force write to physical disk
- **Battery-backed cache**: Survive power failure

---

## BASE Properties

**BASE** stands for: **Basically Available, Soft state, Eventual consistency**

These properties describe the characteristics of many NoSQL databases that prioritize availability and partition tolerance over strong consistency.

### **BA - Basically Available**

**Definition**: The system guarantees availability (responds to queries) but may return stale or incomplete data.

\`\`\`
E-commerce site during Black Friday:
- 1000 requests per second
- Database replica temporarily offline
- System continues serving requests from other replicas
- Some users see slightly stale product prices (1-2 seconds old)
- Better than: "Site unavailable, try again later"
\`\`\`

**Contrast with ACID:**
- ACID: Unavailable during failures (to maintain consistency)
- BASE: Available even with partial failures (eventual consistency)

---

### **S - Soft State**

**Definition**: The state of the system may change over time, even without input, due to eventual consistency.

\`\`\`
Soft state example:
Time 10:00 - User updates profile picture
Time 10:01 - Query US server: Sees new picture
Time 10:02 - Query EU server: Sees old picture (replication lag)
Time 10:05 - Query EU server: Sees new picture (replicated)

State "changed" from old to new picture without user action
\`\`\`

**Contrast with ACID:**
- ACID: Hard state (deterministic, doesn't change without transaction)
- BASE: Soft state (may change as replication propagates)

---

### **E - Eventual Consistency**

**Definition**: Given enough time without updates, all replicas will converge to the same state.

\`\`\`
User posts tweet:
- Written to primary datacenter (US East)
- Replication to other datacenters (EU, Asia, US West)
- Users in different regions may see tweet at different times
- Eventually (typically < 1 second), all see the tweet
\`\`\`

**Guarantee**: Eventually consistent, no time bound

**Acceptable for**: Social media, product catalogs, analytics
**Not acceptable for**: Banking, inventory, reservations

---

## ACID vs BASE Comparison

| Aspect | ACID | BASE |
|--------|------|------|
| **Consistency** | Strong (immediate) | Eventual (converges over time) |
| **Availability** | Lower (during failures) | Higher (always responsive) |
| **Isolation** | Full isolation | Weak or no isolation |
| **Durability** | Guaranteed immediately | Eventually durable |
| **Scalability** | Vertical (scale up) | Horizontal (scale out) |
| **Use Cases** | Banking, e-commerce transactions | Social media, content delivery, analytics |
| **Databases** | PostgreSQL, MySQL, Oracle | Cassandra, DynamoDB, Riak, MongoDB |
| **CAP Choice** | CP (Consistency + Partition tolerance) | AP (Availability + Partition tolerance) |

---

## When to Use ACID

### **✅ Use ACID When:**

**1. Financial Transactions**
- Banking: Money transfers must be atomic
- E-commerce: Order + payment + inventory must succeed together
- Accounting: Double-entry bookkeeping requires consistency

**2. Inventory Management**
- Cannot oversell products
- Stock levels must be accurate
- Reservations must be atomic

**3. Data Integrity Critical**
- Healthcare records
- Legal documents
- Audit logs

**4. Multi-Step Operations Must Succeed Together**
- User registration: Create account + send verification + log audit
- All steps must succeed or rollback

**Example: E-commerce Checkout**

\`\`\`sql
BEGIN TRANSACTION;
  -- Check inventory
  SELECT quantity FROM products WHERE id = 'P123' FOR UPDATE;
  -- quantity = 5

  -- Deduct inventory
  UPDATE products SET quantity = quantity - 1 WHERE id = 'P123';

  -- Create order
  INSERT INTO orders (user_id, product_id, amount) VALUES (1, 'P123', 29.99);

  -- Charge payment
  INSERT INTO payments (order_id, amount, status) VALUES (LAST_INSERT_ID(), 29.99, 'charged');

  -- If any step fails, ROLLBACK all
COMMIT;
\`\`\`

**Why ACID**: Ensures no inventory deducted without order, no order without payment

---

## When to Use BASE

### **✅ Use BASE When:**

**1. High Availability Required**
- Social media must always be accessible
- Content sites must always serve articles
- Better to show slightly stale data than be down

**2. Scale is Critical**
- Billions of records
- Millions of requests per second
- Geographic distribution required

**3. Slight Staleness Acceptable**
- Like counts on social media
- Product reviews
- Analytics dashboards
- User profiles

**4. Independent Operations**
- Logging events
- Tracking user activity
- Collecting metrics
- Each operation independent, no multi-step atomicity needed

**Example: Social Media Feed**

\`\`\`
User posts photo:
1. Write to Cassandra (primary datacenter)
2. Acknowledge to user immediately
3. Replicate to other datacenters asynchronously
4. Some users see photo after 1-2 seconds (eventual consistency)

Why BASE:
- High availability (users can always post)
- Scales to billions of posts
- 1-2 second staleness acceptable for social feed
- No atomic multi-step operation needed
\`\`\`

---

## Hybrid Approaches

Modern systems often use **both ACID and BASE** for different data:

### **Example: E-commerce Platform**

\`\`\`
ACID (PostgreSQL):
- User accounts
- Orders
- Payments
- Inventory
→ Strong consistency required

BASE (Cassandra):
- Product catalog
- User reviews
- Search history
- Activity logs
→ High availability, eventual consistency acceptable

Cache (Redis):
- Session storage
- Product page cache
- Shopping cart (temporary)
→ Fastest, can lose some data
\`\`\`

**Why Hybrid**: Use the right tool for each data type's requirements

---

## ACID in Distributed Systems

Achieving ACID in distributed systems is challenging:

### **Challenge 1: Distributed Transactions**

**Two-Phase Commit (2PC):**

\`\`\`
Coordinator asks all participants: "Can you commit?"
All respond: "Yes" or "No"
If all "Yes": Coordinator tells all to commit
If any "No": Coordinator tells all to rollback

Problem:
- Blocking protocol (participants wait for coordinator)
- Coordinator failure = participants blocked indefinitely
- Not partition tolerant
\`\`\`

**Saga Pattern (Alternative):**

\`\`\`
Instead of single atomic transaction:
- Execute series of local transactions
- If one fails, execute compensating transactions (undo)

Example: Book flight + hotel
1. Book flight (commit)
2. Book hotel (commit)
3. Hotel fails: Execute compensating transaction (cancel flight)

Trade-off: Not truly atomic, but more available
\`\`\`

### **Challenge 2: CAP Theorem**

**ACID databases are typically CP (Consistency + Partition tolerance)**
- During partition, sacrifice availability for consistency
- Can't serve requests without guaranteeing consistency

**BASE databases are typically AP (Availability + Partition tolerance)**
- During partition, sacrifice consistency for availability
- Serve requests with stale data, converge later

---

## Real-World Examples

### **ACID: Stripe Payment Processing**

\`\`\`
Requirements:
- Never double-charge customer
- Never lose payment record
- Money transfer must be atomic

Database: PostgreSQL (ACID)

Why:
- Atomicity: Charge + record must happen together
- Consistency: Balance constraints must be enforced
- Isolation: Concurrent payments don't interfere
- Durability: Payment recorded even if system crashes
\`\`\`

### **BASE: Netflix Content Catalog**

\`\`\`
Requirements:
- Always available (must serve movie listings)
- Scales to millions of users
- Slight staleness acceptable (new movie appears after 1 minute is fine)

Database: Cassandra (BASE)

Why:
- Basically available: Always serves content even during failures
- Soft state: Movie catalog may be slightly different across regions
- Eventual consistency: New movies propagate to all regions eventually
\`\`\`

---

## Interview Tips

### **Question**: "ACID or BASE for your system?"

**Bad Answer**: "I'll use NoSQL because it's web scale."

**Good Answer**: "For this system, I'd use ACID (PostgreSQL) for order and payment data because we cannot risk double-charging users or losing orders - atomicity and consistency are critical. For the product catalog and user reviews, I'd use BASE (DynamoDB) because slight staleness is acceptable and we need high availability. This hybrid approach gives us strong guarantees where needed and scalability where appropriate."

### **Show Depth:**

- Explain specific ACID properties needed (atomicity for multi-step transactions)
- Discuss isolation levels (READ COMMITTED vs SERIALIZABLE)
- Mention real-world systems (Stripe uses PostgreSQL, Netflix uses Cassandra)
- Relate to CAP theorem (ACID = CP, BASE = AP)

---

## Key Takeaways

1. **ACID**: Atomicity, Consistency, Isolation, Durability - strong guarantees for reliable transactions
2. **BASE**: Basically Available, Soft state, Eventual consistency - high availability over consistency
3. **Isolation levels**: Read Uncommitted → Read Committed → Repeatable Read → Serializable
4. **Choose ACID for**: Financial transactions, inventory, multi-step operations requiring atomicity
5. **Choose BASE for**: Social media, content delivery, analytics, high-scale systems
6. **Hybrid approach**: Use both ACID and BASE for different data types in same system
7. **ACID in distributed systems**: Challenging (2PC, Saga pattern)
8. **CAP relation**: ACID typically CP, BASE typically AP
9. **Trade-offs**: Consistency ↔ Availability ↔ Scalability
10. **In interviews**: Justify choice based on specific requirements

---

## Summary

ACID and BASE represent two philosophies for database design. ACID prioritizes **correctness and consistency** through atomicity, consistency, isolation, and durability - ideal for financial transactions and systems requiring strong guarantees. BASE prioritizes **availability and scalability** through eventual consistency - ideal for high-scale, high-availability systems where slight staleness is acceptable. Modern systems often use both, applying ACID properties to critical transactional data and BASE properties to high-volume, read-heavy data. Understanding the trade-offs allows you to choose the right approach for each part of your system.`,
};
