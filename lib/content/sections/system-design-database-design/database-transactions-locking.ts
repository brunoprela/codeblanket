/**
 * Database Transactions & Locking Section
 */

export const databasetransactionslockingSection = {
  id: 'database-transactions-locking',
  title: 'Database Transactions & Locking',
  content: `Database transactions and locking mechanisms are fundamental to maintaining data consistency in concurrent systems. Understanding how transactions work, isolation levels, and various locking strategies is critical for building reliable applications.

## What is a Transaction?

A **transaction** is a sequence of database operations that are treated as a single unit of work. Transactions follow the **ACID** properties we covered earlier.

### Transaction Example:

\`\`\`sql
-- Bank transfer: $100 from Account A to Account B
BEGIN TRANSACTION;

-- Deduct from Account A
UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';

-- Add to Account B
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';

-- Check constraints are satisfied
IF (SELECT balance FROM accounts WHERE account_id = 'A') < 0 THEN
    ROLLBACK;  -- Undo all changes
ELSE
    COMMIT;    -- Make changes permanent
END IF;
\`\`\`

**Key Operations:**
- \`BEGIN/START TRANSACTION\`: Start a transaction
- \`COMMIT\`: Make all changes permanent
- \`ROLLBACK\`: Undo all changes
- \`SAVEPOINT\`: Create a checkpoint within transaction

## Why Transactions Matter

**Without Transactions (Broken):**
\`\`\`sql
-- Step 1: Deduct from A
UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
-- ⚡ Application crashes here

-- Step 2: Never executes
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';

-- Result: $100 disappeared from system!
\`\`\`

**With Transactions (Safe):**
\`\`\`sql
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
-- ⚡ Crash here → automatic ROLLBACK
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';
COMMIT;

-- Result: Either both updates or neither (atomicity)
\`\`\`

## Concurrency Problems

When multiple transactions run concurrently, several anomalies can occur:

### 1. Dirty Read

**Problem:** Transaction reads uncommitted changes from another transaction.

\`\`\`
Transaction A                   Transaction B
BEGIN                           BEGIN
UPDATE accounts                 
  SET balance = 500             
  WHERE id = 1                  
                                SELECT balance FROM accounts WHERE id = 1
                                -- Reads 500 (uncommitted)
ROLLBACK                        
-- Balance is actually 1000     -- But B thinks it's 500 (dirty read!)
\`\`\`

**Impact:** B makes decisions based on data that never existed.

### 2. Non-Repeatable Read

**Problem:** Transaction reads same row twice and gets different values.

\`\`\`
Transaction A                   Transaction B
BEGIN                           BEGIN
SELECT balance FROM accounts    
  WHERE id = 1                  
-- Reads 1000                   
                                UPDATE accounts SET balance = 500 WHERE id = 1
                                COMMIT
SELECT balance FROM accounts    
  WHERE id = 1                  
-- Reads 500                    
-- Same query, different result!
\`\`\`

**Impact:** Inconsistent reads within a transaction.

### 3. Phantom Read

**Problem:** Transaction reads a set of rows twice and gets different row counts.

\`\`\`
Transaction A                   Transaction B
BEGIN                           BEGIN
SELECT COUNT(*) FROM orders     
  WHERE user_id = 123           
-- Returns 5                    
                                INSERT INTO orders (user_id, ...) VALUES (123, ...)
                                COMMIT
SELECT COUNT(*) FROM orders     
  WHERE user_id = 123           
-- Returns 6 (phantom row appeared!)
\`\`\`

**Impact:** Aggregations and range queries yield inconsistent results.

### 4. Lost Update

**Problem:** Two transactions update the same row; one update is lost.

\`\`\`
Transaction A                   Transaction B
BEGIN                           BEGIN
SELECT balance FROM accounts    SELECT balance FROM accounts
  WHERE id = 1                    WHERE id = 1
-- Reads 1000                   -- Reads 1000

UPDATE accounts                 UPDATE accounts
  SET balance = 1000 + 100        SET balance = 1000 + 50
  WHERE id = 1                    WHERE id = 1
COMMIT (balance = 1100)         COMMIT (balance = 1050)

-- Result: B overwrites A's update, losing $100!
\`\`\`

**Impact:** Data updates are silently lost.

## Isolation Levels

**Isolation levels** define how transactions are isolated from each other, trading off consistency for performance.

### Isolation Levels (Weakest → Strongest)

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|-------|------------|---------------------|--------------|-------------|
| Read Uncommitted | ✗ Possible | ✗ Possible | ✗ Possible | Fastest |
| Read Committed | ✓ Prevented | ✗ Possible | ✗ Possible | Fast |
| Repeatable Read | ✓ Prevented | ✓ Prevented | ✗ Possible | Slower |
| Serializable | ✓ Prevented | ✓ Prevented | ✓ Prevented | Slowest |

### 1. Read Uncommitted

**Behavior:** Can read uncommitted changes from other transactions (dirty reads).

\`\`\`sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
BEGIN TRANSACTION;
SELECT * FROM accounts WHERE account_id = 'A';
-- Might read uncommitted data!
COMMIT;
\`\`\`

**Use Cases:**
- ✅ Approximate analytics (count, sum)
- ✅ Dashboards where exact numbers don't matter
- ❌ Any critical application logic

**Example:**
\`\`\`sql
-- Dashboard: "~10,000 active users"
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SELECT COUNT(*) FROM users WHERE status = 'active';
-- Fast, but might be slightly off
\`\`\`

### 2. Read Committed (Default in most databases)

**Behavior:** Only reads committed data. Each query sees latest committed data.

\`\`\`sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION;
SELECT balance FROM accounts WHERE account_id = 'A';  -- Reads 1000
-- Another transaction commits, changing balance to 1100
SELECT balance FROM accounts WHERE account_id = 'A';  -- Reads 1100 (non-repeatable read)
COMMIT;
\`\`\`

**Use Cases:**
- ✅ Most OLTP applications
- ✅ Web applications (each request is a transaction)
- ✅ Default for PostgreSQL, SQL Server, Oracle

**Example:**
\`\`\`sql
-- E-commerce: Check product availability
BEGIN;
SELECT stock_quantity FROM products WHERE product_id = 123;
-- If quantity > 0, proceed with order
-- Another transaction might buy the product between queries
COMMIT;
\`\`\`

### 3. Repeatable Read

**Behavior:** Sees a consistent snapshot of data throughout the transaction.

\`\`\`sql
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN TRANSACTION;
SELECT balance FROM accounts WHERE account_id = 'A';  -- Reads 1000
-- Another transaction commits, changing balance to 1100
SELECT balance FROM accounts WHERE account_id = 'A';  -- Still reads 1000 (repeatable)
COMMIT;
\`\`\`

**Use Cases:**
- ✅ Multi-step workflows requiring consistency
- ✅ Financial calculations
- ✅ Report generation
- ✅ Default for MySQL InnoDB

**Example:**
\`\`\`sql
-- Financial report: Calculate account summary
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT SUM(balance) FROM accounts WHERE user_id = 123;
-- Even if deposits happen during transaction, sum stays consistent
SELECT COUNT(*) FROM accounts WHERE user_id = 123;
COMMIT;
\`\`\`

**PostgreSQL Implementation:**
- Uses MVCC (Multi-Version Concurrency Control)
- Each transaction sees a snapshot from transaction start time
- No locking for reads (readers don't block writers)

**MySQL InnoDB Implementation:**
- Also uses MVCC
- Prevents non-repeatable reads
- Still allows phantom reads (in some scenarios)

### 4. Serializable (Strictest)

**Behavior:** Transactions execute as if they ran serially, one after another.

\`\`\`sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN TRANSACTION;
SELECT SUM(balance) FROM accounts WHERE user_id = 123;
-- No other transaction can modify these rows until this commits
COMMIT;
\`\`\`

**Use Cases:**
- ✅ Financial systems (banking, payments)
- ✅ Inventory management (prevent overselling)
- ✅ Ticket booking (prevent double-booking)

**Example:**
\`\`\`sql
-- Ticket booking: Prevent double-booking
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;

-- Check if seat is available
SELECT status FROM seats WHERE seat_id = 'A1' FOR UPDATE;

-- If available, book it
UPDATE seats SET status = 'booked', user_id = 123 WHERE seat_id = 'A1';

COMMIT;
-- No other transaction can book A1 concurrently
\`\`\`

**Performance Impact:**
- Highest consistency guarantee
- Significant performance penalty (serialization, locking)
- Can cause more deadlocks and transaction aborts

**PostgreSQL Serializable Implementation:**
- Uses Serializable Snapshot Isolation (SSI)
- Detects conflicts and aborts transactions
- May see "could not serialize access" errors

## Database Locking

Locks prevent concurrent transactions from interfering with each other.

### Lock Types

#### 1. Shared Lock (S Lock, Read Lock)

**Purpose:** Allows multiple transactions to read, but blocks writes.

\`\`\`sql
-- Acquire shared lock
SELECT * FROM products WHERE product_id = 123 FOR SHARE;
-- Other transactions can also read (shared lock)
-- But cannot write (blocked by shared lock)
\`\`\`

**Use Case:**
\`\`\`sql
-- Multiple users viewing product details (read-only)
BEGIN;
SELECT * FROM products WHERE product_id = 123 FOR SHARE;
-- Display product info
COMMIT;
\`\`\`

#### 2. Exclusive Lock (X Lock, Write Lock)

**Purpose:** Blocks both reads and writes from other transactions.

\`\`\`sql
-- Acquire exclusive lock
SELECT * FROM products WHERE product_id = 123 FOR UPDATE;
-- No other transaction can read or write this row
\`\`\`

**Use Case:**
\`\`\`sql
-- Update inventory: prevent concurrent modifications
BEGIN;
SELECT stock_quantity FROM products WHERE product_id = 123 FOR UPDATE;
-- Exclusive lock acquired

-- Check and update
IF stock_quantity > 0 THEN
    UPDATE products SET stock_quantity = stock_quantity - 1 WHERE product_id = 123;
END IF;

COMMIT;
\`\`\`

### Lock Granularity

**Row-Level Locks:**
\`\`\`sql
-- Lock specific rows
UPDATE products SET price = 29.99 WHERE product_id = 123;
-- Only row 123 is locked
\`\`\`

**Table-Level Locks:**
\`\`\`sql
-- Lock entire table
LOCK TABLE products IN EXCLUSIVE MODE;
-- No other transaction can access table
\`\`\`

**Page-Level Locks:**
- Some databases (SQL Server) lock data pages (8KB blocks)
- Trade-off between row and table locks

**Comparison:**

| Granularity | Concurrency | Overhead | Use Case |
|-------------|-------------|----------|----------|
| Row | High | High | OLTP (frequent small updates) |
| Page | Medium | Medium | Mixed workloads |
| Table | Low | Low | Batch operations, DDL |

### Explicit Locking

**FOR UPDATE:**
\`\`\`sql
BEGIN;
SELECT * FROM orders WHERE order_id = 123 FOR UPDATE;
-- Exclusive lock: only this transaction can modify
UPDATE orders SET status = 'shipped' WHERE order_id = 123;
COMMIT;
\`\`\`

**FOR SHARE:**
\`\`\`sql
BEGIN;
SELECT * FROM orders WHERE order_id = 123 FOR SHARE;
-- Shared lock: others can read, but not modify
-- Read order details...
COMMIT;
\`\`\`

**SKIP LOCKED (PostgreSQL, MySQL 8+):**
\`\`\`sql
-- Job queue: Get next available job (skip locked ones)
BEGIN;
SELECT * FROM jobs 
WHERE status = 'pending' 
ORDER BY created_at 
LIMIT 1 
FOR UPDATE SKIP LOCKED;

-- Process job...
UPDATE jobs SET status = 'processing' WHERE job_id = ...;
COMMIT;
\`\`\`

**Use Case:** Multiple workers processing a job queue without contention.

**NOWAIT:**
\`\`\`sql
-- Fail immediately if row is locked (don't wait)
BEGIN;
SELECT * FROM seats WHERE seat_id = 'A1' FOR UPDATE NOWAIT;
-- If locked, throws error immediately
\`\`\`

**Use Case:** Real-time seat booking (fail fast if unavailable).

## Deadlocks

**Deadlock:** Two or more transactions waiting for each other, creating a cycle.

### Deadlock Example:

\`\`\`
Transaction A                   Transaction B
BEGIN                           BEGIN
UPDATE accounts                 UPDATE accounts
  SET balance = 1000              SET balance = 2000
  WHERE id = 1                    WHERE id = 2
-- Acquires lock on row 1       -- Acquires lock on row 2

UPDATE accounts                 UPDATE accounts
  SET balance = 2000              SET balance = 1000
  WHERE id = 2                    WHERE id = 1
-- Waits for B's lock on row 2  -- Waits for A's lock on row 1

-- DEADLOCK! Both waiting forever
\`\`\`

**Database Response:**
- Detects deadlock
- Aborts one transaction (deadlock victim)
- Other transaction proceeds

**Error Message:**
\`\`\`
ERROR: deadlock detected
DETAIL: Process 1234 waits for ShareLock on transaction 5678;
        blocked by process 5678.
        Process 5678 waits for ShareLock on transaction 1234;
        blocked by process 1234.
\`\`\`

### Preventing Deadlocks

**1. Consistent Lock Order:**
\`\`\`sql
-- BAD: Different order
Transaction A: Lock account 1, then 2
Transaction B: Lock account 2, then 1  -- Can deadlock!

-- GOOD: Same order
Transaction A: Lock account 1, then 2
Transaction B: Lock account 1, then 2  -- No deadlock possible
\`\`\`

**2. Keep Transactions Short:**
\`\`\`sql
-- BAD: Long transaction
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- ... complex business logic, API calls, etc.
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- GOOD: Short transaction
-- Do business logic outside transaction
calculated_amount = complex_calculation()

BEGIN;
UPDATE accounts SET balance = balance - calculated_amount WHERE id = 1;
UPDATE accounts SET balance = balance + calculated_amount WHERE id = 2;
COMMIT;  -- Locks held for minimal time
\`\`\`

**3. Use Lower Isolation Levels:**
- Read Committed reduces locking compared to Serializable
- Trade-off: May need to handle race conditions in application

**4. Timeout and Retry:**
\`\`\`python
def transfer_with_retry (from_account, to_account, amount, max_retries=3):
    for attempt in range (max_retries):
        try:
            with db.transaction():
                db.execute("UPDATE accounts SET balance = balance - %s WHERE id = %s", 
                          (amount, from_account))
                db.execute("UPDATE accounts SET balance = balance + %s WHERE id = %s", 
                          (amount, to_account))
                return True
        except DeadlockDetected:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    return False
\`\`\`

## Optimistic vs Pessimistic Locking

### Pessimistic Locking

**Strategy:** Assume conflicts will happen; acquire locks upfront.

\`\`\`sql
BEGIN;
-- Lock immediately
SELECT * FROM products WHERE product_id = 123 FOR UPDATE;

-- Do work (locks held)
new_quantity = current_quantity - 1

UPDATE products SET stock_quantity = new_quantity WHERE product_id = 123;
COMMIT;
\`\`\`

**Pros:**
- Guaranteed to succeed (no conflicts)
- Simple to reason about

**Cons:**
- Reduced concurrency (locks block others)
- Can cause deadlocks
- Performance impact

**Use Cases:**
- High contention scenarios
- Financial transactions
- Inventory management

### Optimistic Locking

**Strategy:** Assume conflicts are rare; detect conflicts at commit time.

\`\`\`sql
-- Read without locks
SELECT product_id, stock_quantity, version FROM products WHERE product_id = 123;
-- version = 5

-- Do work (no locks held)
new_quantity = stock_quantity - 1

-- Update with version check
UPDATE products 
SET stock_quantity = new_quantity, version = version + 1 
WHERE product_id = 123 AND version = 5;

-- Check affected rows
IF affected_rows = 0 THEN
    -- Someone else updated it; conflict detected
    ROLLBACK;
    RETRY;
ELSE
    COMMIT;
END IF;
\`\`\`

**Pros:**
- High concurrency (no locks)
- Better performance for low-contention scenarios
- No deadlocks

**Cons:**
- Transactions may fail and need retry
- More complex application logic
- Inefficient for high-contention scenarios

**Use Cases:**
- Web applications (low contention per record)
- RESTful APIs
- Read-heavy workloads with occasional writes

**Version Column Pattern:**
\`\`\`sql
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(255),
    stock_quantity INT,
    version INT DEFAULT 0  -- Optimistic lock version
);

-- Application code
def update_product (product_id, new_data):
    product = db.query("SELECT * FROM products WHERE product_id = %s", product_id)
    
    # User modifies data...
    
    # Update with version check
    result = db.execute("""
        UPDATE products 
        SET name = %s, stock_quantity = %s, version = version + 1 
        WHERE product_id = %s AND version = %s
    """, (new_data.name, new_data.stock, product_id, product.version))
    
    if result.rowcount == 0:
        raise ConcurrentModificationError("Product was modified by another user")
\`\`\`

### Comparison

| Aspect | Pessimistic | Optimistic |
|--------|-------------|------------|
| Locking | Upfront | None |
| Concurrency | Lower | Higher |
| Conflicts | Prevented | Detected |
| Retry Logic | Not needed | Required |
| Best For | High contention | Low contention |
| Deadlocks | Possible | No |

## Real-World Patterns

### 1. E-commerce Inventory

**Problem:** Prevent overselling limited stock.

**Pessimistic Approach:**
\`\`\`sql
BEGIN;
-- Lock product row
SELECT stock_quantity FROM products WHERE product_id = 123 FOR UPDATE;

IF stock_quantity >= order_quantity THEN
    UPDATE products SET stock_quantity = stock_quantity - order_quantity 
    WHERE product_id = 123;
    -- Create order...
    COMMIT;
ELSE
    ROLLBACK;
    RAISE 'Out of stock';
END IF;
\`\`\`

**Optimistic Approach:**
\`\`\`sql
-- Check stock (no lock)
SELECT stock_quantity FROM products WHERE product_id = 123;

-- Create order...

-- Atomic decrement with check
UPDATE products 
SET stock_quantity = stock_quantity - order_quantity 
WHERE product_id = 123 AND stock_quantity >= order_quantity;

IF affected_rows = 0 THEN
    ROLLBACK;
    RAISE 'Out of stock or concurrent update';
ELSE
    COMMIT;
END IF;
\`\`\`

### 2. Seat Booking System

**Problem:** Prevent double-booking seats.

\`\`\`sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;

-- Check seat availability
SELECT status FROM seats WHERE seat_id = 'A1' FOR UPDATE;

IF status = 'available' THEN
    UPDATE seats SET status = 'booked', user_id = 123 WHERE seat_id = 'A1';
    -- Create booking record...
    COMMIT;
ELSE
    ROLLBACK;
    RAISE 'Seat already booked';
END IF;
\`\`\`

**Alternative (NOWAIT):**
\`\`\`sql
BEGIN;

-- Try to lock seat; fail fast if unavailable
SELECT status FROM seats WHERE seat_id = 'A1' FOR UPDATE NOWAIT;

-- If we reach here, we have the lock
UPDATE seats SET status = 'booked', user_id = 123 WHERE seat_id = 'A1';
COMMIT;

-- Handle lock timeout in application
EXCEPTION WHEN lock_not_available THEN
    RAISE 'Seat is being booked by another user';
\`\`\`

### 3. Job Queue (Multiple Workers)

**Problem:** Multiple workers processing jobs without conflicts.

\`\`\`sql
-- Worker pulls next available job
BEGIN;

SELECT * FROM jobs 
WHERE status = 'pending' 
ORDER BY priority DESC, created_at ASC 
LIMIT 1 
FOR UPDATE SKIP LOCKED;

-- Update job status
UPDATE jobs SET status = 'processing', worker_id = current_worker 
WHERE job_id = selected_job_id;

COMMIT;

-- Process job outside transaction...

-- Mark complete
UPDATE jobs SET status = 'completed' WHERE job_id = selected_job_id;
\`\`\`

**SKIP LOCKED** ensures workers don't wait for each other.

### 4. Distributed Counter (High Contention)

**Problem:** Many concurrent increments to a counter.

**Naive (Slow):**
\`\`\`sql
-- High contention on single row
UPDATE counters SET count = count + 1 WHERE counter_id = 'page_views';
\`\`\`

**Optimized (Sharding):**
\`\`\`sql
-- Shard counter into multiple rows
CREATE TABLE counters (
    counter_id VARCHAR(50),
    shard_id INT,
    count BIGINT,
    PRIMARY KEY (counter_id, shard_id)
);

-- Increment random shard (reduces contention)
UPDATE counters 
SET count = count + 1 
WHERE counter_id = 'page_views' AND shard_id = RANDOM_INT(0, 9);

-- Read total (sum all shards)
SELECT SUM(count) FROM counters WHERE counter_id = 'page_views';
\`\`\`

## Interview Tips

**Q: "Explain isolation levels and when to use each."**
- **Read Committed:** Default for most apps; prevents dirty reads
- **Repeatable Read:** Financial reports, multi-step calculations needing consistency
- **Serializable:** Critical operations where any anomaly is unacceptable (banking, booking)

**Q: "How do you prevent deadlocks?"**
1. Acquire locks in consistent order
2. Keep transactions short
3. Use lower isolation levels when possible
4. Implement retry logic with exponential backoff
5. Use NOWAIT or lock timeout to fail fast

**Q: "Optimistic vs Pessimistic locking?"**
- **Pessimistic:** High contention, critical data (inventory, bookings)
- **Optimistic:** Low contention, web apps, better concurrency
- **Trade-off:** Pessimistic reduces concurrency but guarantees success; optimistic allows higher concurrency but requires retry logic

**Q: "Design a transaction for X"**
- Identify what needs to be atomic
- Choose appropriate isolation level
- Consider locking strategy (FOR UPDATE, FOR SHARE)
- Handle conflicts and retries
- Keep transaction scope minimal

## Key Takeaways

1. **Transactions ensure ACID properties** (atomicity, consistency, isolation, durability)
2. **Isolation levels trade consistency for performance** (Read Committed → Serializable)
3. **Read Committed is default and suitable for most applications**
4. **Serializable provides strongest guarantees but lowest performance**
5. **Locks prevent conflicts: Shared (read) vs Exclusive (write)**
6. **Row-level locks enable high concurrency; table locks for batch operations**
7. **Deadlocks occur when transactions wait in a cycle; database auto-resolves by aborting one**
8. **Prevent deadlocks: consistent lock order, short transactions, retry logic**
9. **Pessimistic locking: acquire locks early (high contention)**
10. **Optimistic locking: detect conflicts at commit (low contention, higher concurrency)**

## Summary

Transactions provide atomicity and consistency guarantees through ACID properties. Isolation levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable) define how transactions see concurrent changes, trading consistency for performance. Locking mechanisms (shared, exclusive, row-level, table-level) prevent conflicts but can cause deadlocks. Pessimistic locking acquires locks upfront (lower concurrency, guaranteed success), while optimistic locking detects conflicts at commit time (higher concurrency, requires retry logic). Choose strategies based on contention level and consistency requirements.
`,
};
