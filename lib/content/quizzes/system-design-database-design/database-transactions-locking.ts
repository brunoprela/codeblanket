/**
 * Quiz questions for Database Transactions & Locking section
 */

export const databasetransactionslockingQuiz = [
  {
    id: 'trans-disc-1',
    question:
      'Design a transaction strategy for a bank transfer system that needs to transfer money between accounts, maintain audit logs, check daily transfer limits, and handle potential failures. Discuss isolation level, locking strategy, error handling, and how to prevent common issues like deadlocks and lost updates.',
    sampleAnswer: `Comprehensive transaction strategy for bank transfer system:

**Requirements:**1. Transfer funds atomically (all or nothing)
2. Maintain audit trail
3. Enforce daily transfer limits
4. Handle concurrent transfers
5. Prevent deadlocks
6. Ensure no money is lost or created

**Database Schema:**

\`\`\`sql
CREATE TABLE accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    balance DECIMAL(15,2) NOT NULL CHECK (balance >= 0),
    version INT DEFAULT 0,  -- For optimistic locking alternative
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE transfer_limits (
    account_id VARCHAR(50) PRIMARY KEY,
    daily_limit DECIMAL(15,2) NOT NULL,
    current_date DATE NOT NULL,
    amount_transferred_today DECIMAL(15,2) DEFAULT 0,
    FOREIGN KEY (account_id) REFERENCES accounts (account_id)
);

CREATE TABLE transfers (
    transfer_id BIGSERIAL PRIMARY KEY,
    from_account VARCHAR(50) NOT NULL,
    to_account VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- pending, completed, failed
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE TABLE audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    transfer_id BIGINT,
    account_id VARCHAR(50),
    action VARCHAR(50),
    old_balance DECIMAL(15,2),
    new_balance DECIMAL(15,2),
    timestamp TIMESTAMP DEFAULT NOW()
);
\`\`\`

**Transaction Strategy:**

\`\`\`python
def transfer_money (from_account_id, to_account_id, amount):
    # Validation outside transaction
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    if from_account_id == to_account_id:
        raise ValueError("Cannot transfer to same account")
    
    # Use Read Committed isolation (default)
    # Sufficient for this use case with explicit locking
    connection.isolation_level = 'READ COMMITTED'
    
    max_retries = 3
    for attempt in range (max_retries):
        try:
            return _execute_transfer (from_account_id, to_account_id, amount)
        except DeadlockDetected as e:
            if attempt == max_retries - 1:
                raise TransferError("Transfer failed after retries due to deadlock")
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        except InsufficientFundsError as e:
            # Don't retry insufficient funds
            raise
    
def _execute_transfer (from_account_id, to_account_id, amount):
    with db.transaction() as tx:
        # Step 1: Create transfer record
        transfer_id = tx.execute("""
            INSERT INTO transfers (from_account, to_account, amount, status)
            VALUES (%s, %s, %s, 'pending')
            RETURNING transfer_id
        """, (from_account_id, to_account_id, amount))[0]
        
        # Step 2: Lock accounts in consistent order (prevent deadlocks)
        accounts_to_lock = sorted([from_account_id, to_account_id])
        
        locked_accounts = {}
        for account_id in accounts_to_lock:
            account = tx.query("""
                SELECT account_id, balance 
                FROM accounts 
                WHERE account_id = %s 
                FOR UPDATE  -- Exclusive lock
            """, (account_id,))[0]
            locked_accounts[account_id] = account
        
        from_account = locked_accounts[from_account_id]
        to_account = locked_accounts[to_account_id]
        
        # Step 3: Check daily transfer limit
        tx.execute("""
            INSERT INTO transfer_limits (account_id, daily_limit, current_date, amount_transferred_today)
            VALUES (%s, 10000, CURRENT_DATE, 0)
            ON CONFLICT (account_id) DO UPDATE
            SET current_date = CASE 
                WHEN transfer_limits.current_date < CURRENT_DATE 
                THEN CURRENT_DATE 
                ELSE transfer_limits.current_date 
            END,
            amount_transferred_today = CASE
                WHEN transfer_limits.current_date < CURRENT_DATE
                THEN 0
                ELSE transfer_limits.amount_transferred_today
            END
        """, (from_account_id,))
        
        limit_info = tx.query("""
            SELECT daily_limit, amount_transferred_today 
            FROM transfer_limits 
            WHERE account_id = %s 
            FOR UPDATE
        """, (from_account_id,))[0]
        
        if limit_info['amount_transferred_today',] + amount > limit_info['daily_limit',]:
            tx.execute("""
                UPDATE transfers 
                SET status = 'failed', error_message = 'Daily limit exceeded'
                WHERE transfer_id = %s
            """, (transfer_id,))
            tx.commit()
            raise DailyLimitExceededError (f"Daily limit exceeded for account {from_account_id}")
        
        # Step 4: Check sufficient funds
        if from_account['balance',] < amount:
            tx.execute("""
                UPDATE transfers 
                SET status = 'failed', error_message = 'Insufficient funds'
                WHERE transfer_id = %s
            """, (transfer_id,))
            tx.commit()
            raise InsufficientFundsError (f"Insufficient funds in account {from_account_id}")
        
        # Step 5: Perform transfer (atomic updates)
        tx.execute("""
            UPDATE accounts 
            SET balance = balance - %s, updated_at = NOW()
            WHERE account_id = %s
        """, (amount, from_account_id))
        
        tx.execute("""
            UPDATE accounts 
            SET balance = balance + %s, updated_at = NOW()
            WHERE account_id = %s
        """, (amount, to_account_id))
        
        # Step 6: Update daily transfer limit counter
        tx.execute("""
            UPDATE transfer_limits 
            SET amount_transferred_today = amount_transferred_today + %s
            WHERE account_id = %s
        """, (amount, from_account_id))
        
        # Step 7: Audit logging
        tx.execute("""
            INSERT INTO audit_log (transfer_id, account_id, action, old_balance, new_balance)
            VALUES 
                (%s, %s, 'debit', %s, %s),
                (%s, %s, 'credit', %s, %s)
        """, (
            transfer_id, from_account_id, from_account['balance',], from_account['balance',] - amount,
            transfer_id, to_account_id, to_account['balance',], to_account['balance',] + amount
        ))
        
        # Step 8: Mark transfer as completed
        tx.execute("""
            UPDATE transfers 
            SET status = 'completed', completed_at = NOW()
            WHERE transfer_id = %s
        """, (transfer_id,))
        
        # Commit transaction
        tx.commit()
        
        return {
            'transfer_id': transfer_id,
            'status': 'completed',
            'from_account': from_account_id,
            'to_account': to_account_id,
            'amount': amount
        }
\`\`\`

**Key Design Decisions:**

**1. Isolation Level: Read Committed**
- Sufficient when combined with explicit FOR UPDATE locks
- Better performance than Repeatable Read or Serializable
- Prevents dirty reads, which is critical for financial data

**2. Lock Order: Sorted Account IDs**
\`\`\`python
accounts_to_lock = sorted([from_account_id, to_account_id])
\`\`\`
- Prevents deadlocks by ensuring all transactions acquire locks in same order
- Transaction A: Lock A1 → Lock A2
- Transaction B: Lock A1 → Lock A2 (same order, no cycle)

**3. Explicit Locking: FOR UPDATE**
- Acquires exclusive locks on account rows
- Prevents concurrent modifications
- Other transactions wait (serialized access to each account)

**4. Error Handling:**
- Validation outside transaction (minimize lock time)
- Retry logic with exponential backoff for deadlocks
- No retry for business logic errors (insufficient funds, limits)
- Failed transfers logged with error messages

**5. Audit Trail:**
- All changes logged atomically within transaction
- If transaction rolls back, audit entries also roll back
- Provides complete history for compliance and debugging

**6. Daily Limits:**
- Reset automatically at day boundary (CURRENT_DATE check)
- Updated atomically with transfer
- Locked to prevent race conditions

**Preventing Common Issues:**

**Deadlocks:**
✅ Consistent lock order (sorted account IDs)
✅ Short transactions (minimal business logic inside transaction)
✅ Retry with exponential backoff

**Lost Updates:**
✅ FOR UPDATE ensures exclusive access
✅ Balance updates are atomic

**Overselling/Overdraft:**
✅ CHECK constraint on balance (balance >= 0)
✅ Explicit balance check before update
✅ Transaction atomicity ensures both checks and updates succeed or fail together

**Race Conditions:**
✅ FOR UPDATE prevents concurrent access to same accounts
✅ Daily limit counter updated atomically

**Performance Optimization:**

\`\`\`python
# 1. Keep transaction short
def transfer_money_optimized (from_account, to_account, amount):
    # Heavy computation OUTSIDE transaction
    validate_accounts (from_account, to_account)
    verify_fraud_rules (from_account, amount)
    
    # Quick transaction INSIDE
    with db.transaction():
        # Lock, check, update, audit
        pass

# 2. Read replicas for read-only queries
def get_account_balance (account_id):
    # Read from replica (no locks)
    return read_replica.query("SELECT balance FROM accounts WHERE account_id = %s", account_id)

# 3. Batch transfers (if applicable)
def batch_transfer (transfers):
    with db.transaction():
        # Lock all accounts once
        # Process all transfers
        # Single commit
        pass
\`\`\`

**Monitoring and Alerts:**

\`\`\`sql
-- Monitor failed transfers
SELECT status, error_message, COUNT(*) 
FROM transfers 
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY status, error_message;

-- Detect frequent deadlocks
SELECT COUNT(*) as deadlock_count
FROM pg_stat_database
WHERE datname = 'bank_db' AND deadlocks > 0;

-- Audit balance consistency
SELECT a.account_id, a.balance,
       COALESCE(SUM(CASE WHEN al.action = 'credit' THEN al.new_balance - al.old_balance ELSE 0 END), 0) -
       COALESCE(SUM(CASE WHEN al.action = 'debit' THEN al.old_balance - al.new_balance ELSE 0 END), 0) as audit_balance
FROM accounts a
LEFT JOIN audit_log al ON a.account_id = al.account_id
GROUP BY a.account_id, a.balance
HAVING a.balance != audit_balance;  -- Detects inconsistencies
\`\`\`

This comprehensive strategy ensures atomic, consistent, and auditable bank transfers while preventing common concurrency issues.`,
    keyPoints: [
      'Use Serializable isolation for financial transactions (strongest guarantees)',
      'Pessimistic locking with consistent lock ordering prevents deadlocks',
      'Comprehensive error handling with exponential backoff retries',
      'Audit logging for compliance and debugging',
      'Idempotency keys prevent duplicate transfers on retries',
    ],
  },
  {
    id: 'trans-disc-2',
    question:
      'Compare optimistic and pessimistic locking strategies for a collaborative document editing system (like Google Docs) where multiple users can edit the same document simultaneously. Which approach would you choose and why? How would you handle conflicts?',
    sampleAnswer: `Comprehensive comparison and solution for collaborative document editing:

**Use Case Analysis:**

*Characteristics:*
- Multiple users editing same document
- High frequency of updates (keystroke-level or paragraph-level)
- Low to medium contention (users often edit different parts)
- Need for near-real-time collaboration
- Conflicts should be minimized but handled gracefully

**Approach 1: Pessimistic Locking (Section-Level)**

\`\`\`sql
-- Schema
CREATE TABLE documents (
    document_id INT PRIMARY KEY,
    title VARCHAR(255),
    created_at TIMESTAMP
);

CREATE TABLE document_sections (
    section_id SERIAL PRIMARY KEY,
    document_id INT,
    section_order INT,
    content TEXT,
    locked_by_user INT,
    locked_at TIMESTAMP,
    version INT DEFAULT 0
);

CREATE TABLE edit_locks (
    lock_id SERIAL PRIMARY KEY,
    section_id INT,
    user_id INT,
    acquired_at TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE (section_id)
);
\`\`\`

**Implementation:**

\`\`\`python
def acquire_section_lock (section_id, user_id, timeout_seconds=30):
    with db.transaction():
        # Try to acquire lock
        result = db.execute("""
            INSERT INTO edit_locks (section_id, user_id, acquired_at, expires_at)
            VALUES (%s, %s, NOW(), NOW() + INTERVAL '%s seconds')
            ON CONFLICT (section_id) DO NOTHING
            RETURNING lock_id
        """, (section_id, user_id, timeout_seconds))
        
        if result.rowcount == 0:
            # Check if lock exists and is expired
            db.execute("""
                DELETE FROM edit_locks 
                WHERE section_id = %s AND expires_at < NOW()
            """, (section_id,))
            
            # Retry acquisition
            result = db.execute("""
                INSERT INTO edit_locks (section_id, user_id, acquired_at, expires_at)
                VALUES (%s, %s, NOW(), NOW() + INTERVAL '%s seconds')
                RETURNING lock_id
            """, (section_id, user_id, timeout_seconds))
            
            if result.rowcount == 0:
                # Lock held by another user
                current_lock = db.query("""
                    SELECT user_id, acquired_at, expires_at 
                    FROM edit_locks 
                    WHERE section_id = %s
                """, (section_id,))[0]
                raise SectionLockedError (f"Section locked by user {current_lock['user_id',]}")
        
        return result[0]['lock_id',]

def update_section_with_lock (section_id, user_id, new_content):
    with db.transaction():
        # Verify user holds lock
        lock = db.query("""
            SELECT * FROM edit_locks 
            WHERE section_id = %s AND user_id = %s AND expires_at > NOW()
        """, (section_id, user_id))
        
        if not lock:
            raise UnauthorizedError("You don't hold the lock for this section")
        
        # Update content
        db.execute("""
            UPDATE document_sections 
            SET content = %s, version = version + 1 
            WHERE section_id = %s
        """, (new_content, section_id))
        
        # Extend lock
        db.execute("""
            UPDATE edit_locks 
            SET expires_at = NOW() + INTERVAL '30 seconds'
            WHERE section_id = %s AND user_id = %s
        """, (section_id, user_id))

def release_section_lock (section_id, user_id):
    db.execute("""
        DELETE FROM edit_locks 
        WHERE section_id = %s AND user_id = %s
    """, (section_id, user_id))
\`\`\`

**Pros:**
✅ Clear ownership (one user edits a section at a time)
✅ No merge conflicts
✅ Simple to implement
✅ Works well for coarse-grained edits (paragraph/section level)

**Cons:**
❌ Reduced collaboration (users blocked from editing locked sections)
❌ Lock management overhead
❌ Dead locks if user disconnects without releasing lock
❌ Poor UX ("Section is locked, try again later")

**Approach 2: Optimistic Locking (Version-Based)**

\`\`\`sql
-- Schema
CREATE TABLE documents (
    document_id INT PRIMARY KEY,
    title VARCHAR(255),
    content TEXT,
    version INT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE document_revisions (
    revision_id SERIAL PRIMARY KEY,
    document_id INT,
    content TEXT,
    version INT,
    created_by INT,
    created_at TIMESTAMP DEFAULT NOW()
);
\`\`\`

**Implementation:**

\`\`\`python
def get_document (document_id):
    return db.query("""
        SELECT document_id, content, version 
        FROM documents 
        WHERE document_id = %s
    """, (document_id,))[0]

def save_document (document_id, new_content, expected_version, user_id):
    with db.transaction():
        # Atomic update with version check
        result = db.execute("""
            UPDATE documents 
            SET content = %s, version = version + 1, updated_at = NOW()
            WHERE document_id = %s AND version = %s
            RETURNING version
        """, (new_content, document_id, expected_version))
        
        if result.rowcount == 0:
            # Version mismatch - someone else updated
            current = db.query("""
                SELECT version, content, updated_at 
                FROM documents 
                WHERE document_id = %s
            """, (document_id,))[0]
            
            raise ConcurrentModificationError({
                'expected_version': expected_version,
                'current_version': current['version',],
                'current_content': current['content',]
            })
        
        # Save revision history
        db.execute("""
            INSERT INTO document_revisions (document_id, content, version, created_by)
            VALUES (%s, %s, %s, %s)
        """, (document_id, new_content, result[0]['version',], user_id))
        
        return result[0]['version',]
\`\`\`

**Conflict Resolution (Client-Side):**

\`\`\`javascript
// Client periodically saves changes
async function saveDocumentWithRetry (documentId, content, currentVersion) {
    try {
        const newVersion = await api.saveDocument (documentId, content, currentVersion);
        return { success: true, version: newVersion };
    } catch (error) {
        if (error instanceof ConcurrentModificationError) {
            // Merge conflicts
            const merged = mergeChanges (content, error.current_content);
            
            // Show diff to user
            showConflictDialog({
                yourChanges: content,
                theirChanges: error.current_content,
                merged: merged
            });
            
            // Retry with merged content
            return saveDocumentWithRetry (documentId, merged, error.current_version);
        }
        throw error;
    }
}

function mergeChanges (localContent, serverContent) {
    // Simple three-way merge (diff-match-patch library)
    const baseContent = lastSavedContent;  // Stored from last successful save
    
    const dmp = new DiffMatchPatch();
    
    // Create patches
    const patches1 = dmp.patch_make (baseContent, localContent);
    const patches2 = dmp.patch_make (baseContent, serverContent);
    
    // Apply both patches
    const [merged, results] = dmp.patch_apply([...patches1, ...patches2], baseContent);
    
    return merged;
}
\`\`\`

**Pros:**
✅ High concurrency (no locks)
✅ Better collaboration (multiple users edit simultaneously)
✅ No lock management overhead
✅ No dead locks

**Cons:**
❌ Conflicts require resolution
❌ More complex client logic
❌ Potential data loss if merge fails
❌ Frequent conflicts with high contention

**Approach 3: Operational Transformation (OT) / CRDT (Recommended)**

*This is what Google Docs actually uses.*

**Concept:**
- Each edit is an operation (insert, delete, retain)
- Operations are transformed to account for concurrent edits
- Eventual consistency: all clients converge to same state

\`\`\`sql
-- Schema
CREATE TABLE documents (
    document_id INT PRIMARY KEY,
    title VARCHAR(255),
    content TEXT
);

CREATE TABLE operations (
    operation_id BIGSERIAL PRIMARY KEY,
    document_id INT,
    user_id INT,
    operation_type VARCHAR(20),  -- insert, delete, retain
    position INT,
    text TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    applied BOOLEAN DEFAULT FALSE
);
\`\`\`

**Implementation (Simplified OT):**

\`\`\`python
# Server: Receive and broadcast operations
def apply_operation (document_id, user_id, operation):
    with db.transaction():
        # Store operation
        op_id = db.execute("""
            INSERT INTO operations (document_id, user_id, operation_type, position, text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING operation_id
        """, (document_id, user_id, operation['type',], operation['position',], operation['text',]))[0]
        
        # Apply to document (simplified)
        current_content = db.query("""
            SELECT content FROM documents WHERE document_id = %s FOR UPDATE
        """, (document_id,))[0]['content',]
        
        if operation['type',] == 'insert':
            new_content = current_content[:operation['position',]] + operation['text',] + current_content[operation['position',]:]
        elif operation['type',] == 'delete':
            new_content = current_content[:operation['position',]] + current_content[operation['position',] + operation['length',]:]
        
        db.execute("""
            UPDATE documents SET content = %s WHERE document_id = %s
        """, (new_content, document_id))
        
        # Broadcast to all connected clients (WebSocket)
        broadcast_operation (document_id, {
            'operation_id': op_id,
            'user_id': user_id,
            'operation': operation
        })
\`\`\`

**Client (OT):**

\`\`\`javascript
// Each client maintains local state
class DocumentEditor {
    constructor (documentId) {
        this.documentId = documentId;
        this.content = "";
        this.pendingOps = [];
        this.lastServerOp = 0;
    }
    
    // User makes local edit
    onLocalEdit (operation) {
        // Apply immediately to local content
        this.content = applyOperation (this.content, operation);
        
        // Queue for server
        this.pendingOps.push (operation);
        
        // Send to server
        this.sendOperation (operation);
    }
    
    // Receive operation from server
    onRemoteOperation (serverOp) {
        // Transform pending operations against server operation
        this.pendingOps = this.pendingOps.map (localOp => 
            transform (localOp, serverOp)
        );
        
        // Apply transformed server operation to local content
        this.content = applyOperation (this.content, serverOp);
        
        this.lastServerOp = serverOp.operation_id;
    }
    
    // Operational Transformation
    function transform (op1, op2) {
        // If both operations are at same position
        if (op1.position === op2.position) {
            // Tie-break by user_id or timestamp
            if (op1.user_id < op2.user_id) {
                return op1;  // op1 goes first
            } else {
                return { ...op1, position: op1.position + op2.text.length };
            }
        }
        
        // If op2 is before op1, shift op1 position
        if (op2.position < op1.position) {
            return { ...op1, position: op1.position + op2.text.length };
        }
        
        return op1;
    }
}
\`\`\`

**Pros:**
✅ True real-time collaboration (like Google Docs)
✅ Automatic conflict resolution
✅ No locks, no blocking
✅ Eventual consistency guaranteed
✅ Works offline (sync when reconnected)

**Cons:**
❌ Complex implementation (OT algorithms are tricky)
❌ Requires persistent WebSocket connection
❌ Operational transformation edge cases
❌ Higher server complexity

**Recommended Solution: Hybrid Approach**

**For Document Editing:**1. **Use OT/CRDT for real-time character-level editing**
   - Libraries: ShareDB, Yjs, Automerge
   - WebSocket for real-time sync
   
2. **Use Pessimistic Locking for structural changes**
   - Renaming document, changing permissions: acquire lock
   - Prevents conflicts on metadata

3. **Use Optimistic Locking as fallback**
   - If WebSocket disconnects, fall back to version-based saves
   - Periodic snapshots with version numbers

**Implementation:**

\`\`\`python
# Document metadata: pessimistic lock
def rename_document (document_id, new_title, user_id):
    with db.transaction():
        db.execute("SELECT * FROM documents WHERE document_id = %s FOR UPDATE", (document_id,))
        db.execute("UPDATE documents SET title = %s WHERE document_id = %s", (new_title, document_id))

# Real-time content: OT via WebSocket
websocket.on('operation', (op) => {
    applyOperationalTransformation (op);
    broadcastToClients (op);
});

# Fallback: periodic save with optimistic locking
def auto_save (document_id, content, version):
    try:
        new_version = save_with_version_check (document_id, content, version);
        return new_version;
    except ConcurrentModificationError:
        # Reload and retry
        fresh_doc = get_document (document_id);
        return auto_save (document_id, merge (content, fresh_doc.content), fresh_doc.version);
\`\`\`

**Conclusion:**

For a **Google Docs-like system**: Use **Operational Transformation (OT)** or **CRDTs** for real-time collaboration, with pessimistic locking for metadata changes and optimistic locking as fallback. This provides the best user experience with automatic conflict resolution and true real-time collaboration.`,
    keyPoints: [
      'Optimistic locking for high-concurrency, low-contention scenarios',
      'Pessimistic locking for high-contention resources (e.g., last item in stock)',
      'Operational Transformation (OT) or CRDTs for real-time collaborative editing',
      'Version columns enable optimistic locking with conflict detection',
      'Hybrid approaches combine multiple strategies for different use cases',
    ],
  },
  {
    id: 'trans-disc-3',
    question:
      'A ticket booking system allows users to search for events, hold seats temporarily during checkout, and complete purchases. Design the transaction and locking strategy to prevent double-booking, handle abandoned carts (held seats that are never purchased), and support high concurrent traffic. Discuss isolation levels, timeouts, and how to balance user experience with system consistency.',
    sampleAnswer: `Comprehensive ticket booking system with transactions and locking:

**Requirements:**1. Prevent double-booking (critical)
2. Allow temporary holds during checkout
3. Release abandoned holds automatically
4. Support high concurrent traffic
5. Good user experience (fast, fair)
6. Handle peak load (ticket sales start)

**Database Schema:**

\`\`\`sql
CREATE TABLE events (
    event_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    venue VARCHAR(255),
    event_date TIMESTAMP,
    total_capacity INT
);

CREATE TABLE seats (
    seat_id SERIAL PRIMARY KEY,
    event_id INT,
    section VARCHAR(50),
    row VARCHAR(10),
    seat_number VARCHAR(10),
    price DECIMAL(10,2),
    status VARCHAR(20) NOT NULL,  -- available, held, booked
    held_by_user INT,
    held_at TIMESTAMP,
    hold_expires_at TIMESTAMP,
    booked_by_user INT,
    booked_at TIMESTAMP,
    UNIQUE (event_id, section, row, seat_number),
    FOREIGN KEY (event_id) REFERENCES events (event_id)
);

CREATE INDEX idx_seats_available ON seats (event_id, status) WHERE status = 'available';
CREATE INDEX idx_seats_expired ON seats (hold_expires_at) WHERE status = 'held';

CREATE TABLE bookings (
    booking_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    event_id INT NOT NULL,
    total_amount DECIMAL(10,2),
    status VARCHAR(20) NOT NULL,  -- pending, confirmed, cancelled
    created_at TIMESTAMP DEFAULT NOW(),
    confirmed_at TIMESTAMP
);

CREATE TABLE booking_seats (
    booking_id INT,
    seat_id INT,
    PRIMARY KEY (booking_id, seat_id),
    FOREIGN KEY (booking_id) REFERENCES bookings (booking_id),
    FOREIGN KEY (seat_id) REFERENCES seats (seat_id)
);
\`\`\`

**Step 1: Search Available Seats (No Locks)**

\`\`\`python
def search_available_seats (event_id, section=None, num_seats=1):
    # Read-only query, no locks needed
    # Use Read Committed isolation (default)
    
    query = """
        SELECT seat_id, section, row, seat_number, price, status
        FROM seats
        WHERE event_id = %s 
          AND status = 'available'
    """
    params = [event_id]
    
    if section:
        query += " AND section = %s"
        params.append (section)
    
    query += " ORDER BY section, row, seat_number LIMIT %s"
    params.append (num_seats * 2)  # Return more options
    
    return db.query (query, params)
\`\`\`

**Step 2: Hold Seats (Optimistic Locking with NOWAIT)**

\`\`\`python
def hold_seats (user_id, seat_ids, hold_duration_minutes=10):
    """
    Attempt to hold seats for user.
    Uses optimistic approach with immediate failure for better UX.
    """
    
    # Clean up expired holds first (background job does this too)
    release_expired_holds()
    
    with db.transaction (isolation_level='READ COMMITTED'):
        held_seats = []
        
        for seat_id in seat_ids:
            try:
                # Try to acquire lock with NOWAIT
                seat = db.query("""
                    SELECT seat_id, status, held_by_user, hold_expires_at
                    FROM seats
                    WHERE seat_id = %s
                    FOR UPDATE NOWAIT
                """, (seat_id,))[0]
                
                # Check if seat is available
                if seat['status',] != 'available':
                    raise SeatUnavailableError (f"Seat {seat_id} is no longer available")
                
                # Hold the seat
                db.execute("""
                    UPDATE seats
                    SET status = 'held',
                        held_by_user = %s,
                        held_at = NOW(),
                        hold_expires_at = NOW() + INTERVAL '%s minutes'
                    WHERE seat_id = %s
                """, (user_id, hold_duration_minutes, seat_id))
                
                held_seats.append (seat_id)
                
            except LockNotAvailable:
                # Seat is being held/booked by another user right now
                # Release already held seats and fail fast
                rollback_holds (held_seats)
                raise SeatUnavailableError (f"Seat {seat_id} is being selected by another user")
        
        # All seats successfully held
        return {
            'held_seats': held_seats,
            'expires_at': datetime.now() + timedelta (minutes=hold_duration_minutes),
            'hold_duration_seconds': hold_duration_minutes * 60
        }

def rollback_holds (seat_ids):
    """Release seats if partial hold fails"""
    db.execute("""
        UPDATE seats
        SET status = 'available',
            held_by_user = NULL,
            held_at = NULL,
            hold_expires_at = NULL
        WHERE seat_id = ANY(%s)
    """, (seat_ids,))
\`\`\`

**Step 3: Complete Purchase (Pessimistic Locking)**

\`\`\`python
def complete_purchase (user_id, seat_ids, payment_token):
    """
    Convert held seats to confirmed booking.
    Uses FOR UPDATE to ensure seats are still held by this user.
    """
    
    with db.transaction (isolation_level='SERIALIZABLE'):
        # Verify seats are held by this user and not expired
        seats = db.query("""
            SELECT seat_id, status, held_by_user, hold_expires_at, price
            FROM seats
            WHERE seat_id = ANY(%s)
            FOR UPDATE
        """, (seat_ids,))
        
        for seat in seats:
            if seat['status',] != 'held':
                raise BookingError (f"Seat {seat['seat_id',]} is not held")
            
            if seat['held_by_user',] != user_id:
                raise UnauthorizedError (f"Seat {seat['seat_id',]} is held by another user")
            
            if seat['hold_expires_at',] < datetime.now():
                raise BookingError (f"Hold expired for seat {seat['seat_id',]}")
        
        # Calculate total
        total_amount = sum (seat['price',] for seat in seats)
        
        # Process payment (idempotent, outside DB transaction)
        # In practice, call payment service here
        payment_result = process_payment (user_id, total_amount, payment_token)
        
        if not payment_result['success',]:
            raise PaymentError("Payment failed")
        
        # Create booking
        booking_id = db.execute("""
            INSERT INTO bookings (user_id, event_id, total_amount, status, confirmed_at)
            VALUES (%s, %s, %s, 'confirmed', NOW())
            RETURNING booking_id
        """, (user_id, seats[0]['event_id',], total_amount))[0]['booking_id',]
        
        # Update seats to booked
        db.execute("""
            UPDATE seats
            SET status = 'booked',
                held_by_user = NULL,
                held_at = NULL,
                hold_expires_at = NULL,
                booked_by_user = %s,
                booked_at = NOW()
            WHERE seat_id = ANY(%s)
        """, (user_id, seat_ids))
        
        # Link seats to booking
        for seat_id in seat_ids:
            db.execute("""
                INSERT INTO booking_seats (booking_id, seat_id)
                VALUES (%s, %s)
            """, (booking_id, seat_id))
        
        db.commit()
        
        return {
            'booking_id': booking_id,
            'status': 'confirmed',
            'seats': seats,
            'total_amount': total_amount
        }
\`\`\`

**Step 4: Automatic Hold Expiration (Background Job)**

\`\`\`python
def release_expired_holds():
    """
    Background job runs every 10 seconds.
    Releases expired holds to make seats available again.
    """
    
    with db.transaction():
        result = db.execute("""
            UPDATE seats
            SET status = 'available',
                held_by_user = NULL,
                held_at = NULL,
                hold_expires_at = NULL
            WHERE status = 'held' 
              AND hold_expires_at < NOW()
            RETURNING seat_id
        """)
        
        if result.rowcount > 0:
            released_seats = [row['seat_id',] for row in result]
            logger.info (f"Released {len (released_seats)} expired holds")
            
            # Notify clients via WebSocket
            for seat_id in released_seats:
                websocket_broadcast (f"seat_{seat_id}_available")
        
        return result.rowcount

# Run in background
schedule.every(10).seconds.do (release_expired_holds)
\`\`\`

**Step 5: Extend Hold (During Checkout)**

\`\`\`python
def extend_hold (user_id, seat_ids, additional_minutes=5):
    """
    Allow user to extend hold if they need more time.
    Maximum total hold time: 15 minutes.
    """
    
    with db.transaction():
        result = db.execute("""
            UPDATE seats
            SET hold_expires_at = LEAST(
                hold_expires_at + INTERVAL '%s minutes',
                held_at + INTERVAL '15 minutes'  -- Max 15 min total
            )
            WHERE seat_id = ANY(%s)
              AND status = 'held'
              AND held_by_user = %s
              AND hold_expires_at > NOW()
            RETURNING seat_id, hold_expires_at
        """, (additional_minutes, seat_ids, user_id))
        
        if result.rowcount != len (seat_ids):
            raise BookingError("Some seats could not be extended (expired or not held by you)")
        
        return {
            'extended_seats': [row['seat_id',] for row in result],
            'new_expires_at': result[0]['hold_expires_at',]
        }
\`\`\`

**Step 6: Cancel Hold (Manual Release)**

\`\`\`python
def cancel_hold (user_id, seat_ids):
    """
    User manually releases held seats.
    """
    
    with db.transaction():
        db.execute("""
            UPDATE seats
            SET status = 'available',
                held_by_user = NULL,
                held_at = NULL,
                hold_expires_at = NULL
            WHERE seat_id = ANY(%s)
              AND status = 'held'
              AND held_by_user = %s
        """, (seat_ids, user_id))
\`\`\`

**Handling High Concurrent Traffic:**

**1. Read Replicas for Search:**
\`\`\`python
# Search queries go to read replicas (no locks)
def search_available_seats (event_id):
    return read_replica.query("""
        SELECT seat_id, section, row, seat_number, price
        FROM seats
        WHERE event_id = %s AND status = 'available'
        ORDER BY section, row, seat_number
    """, (event_id,))
\`\`\`

**2. Connection Pooling:**
\`\`\`python
# Configure connection pool
db_pool = ConnectionPool(
    min_connections=10,
    max_connections=100,
    max_idle_time=300,
    max_lifetime=3600
)
\`\`\`

**3. Rate Limiting:**
\`\`\`python
@rate_limit (max_requests=10, window_seconds=60, key=lambda: f"user:{current_user.id}")
def hold_seats (user_id, seat_ids):
    # Prevent user from spamming hold requests
    pass
\`\`\`

**4. Caching (Redis):**
\`\`\`python
# Cache available seat count
def get_available_seats_count (event_id):
    cache_key = f"event:{event_id}:available_count"
    cached = redis.get (cache_key)
    
    if cached is None:
        count = db.query("""
            SELECT COUNT(*) FROM seats 
            WHERE event_id = %s AND status = 'available'
        """, (event_id,))[0]['count',]
        
        redis.setex (cache_key, 10, count)  # Cache for 10 seconds
        return count
    
    return int (cached)
\`\`\`

**5. Queue System for Peak Load:**
\`\`\`python
# On high demand events (Taylor Swift tickets)
if event.is_high_demand():
    # Put user in queue
    queue_position = add_to_queue (user_id, event_id)
    
    # Process queue in order
    process_queue_when_capacity_available()
\`\`\`

**Design Trade-offs:**

| Aspect | Choice | Trade-off |
|--------|--------|-----------|
| Hold Strategy | Temporary holds with expiration | Better UX vs more complex state management |
| Lock Strategy | NOWAIT for holds | Fail fast vs retries |
| Isolation Level | Read Committed for most ops, Serializable for final booking | Performance vs strictest consistency |
| Hold Duration | 10 minutes | User convenience vs seat availability |
| Expiration | Background job every 10s | Near real-time vs server load |

**Monitoring:**

\`\`\`sql
-- Dashboard queries

-- Current holds about to expire
SELECT COUNT(*), MIN(hold_expires_at) - NOW() as time_remaining
FROM seats
WHERE status = 'held' AND hold_expires_at < NOW() + INTERVAL '1 minute';

-- Booking success rate
SELECT 
    COUNT(CASE WHEN status = 'confirmed' THEN 1 END) as confirmed,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as abandoned,
    COUNT(CASE WHEN status = 'confirmed' THEN 1 END) * 100.0 / COUNT(*) as success_rate
FROM bookings
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Contention hotspots
SELECT event_id, COUNT(*) as failed_holds
FROM audit_log
WHERE action = 'hold_failed' AND created_at > NOW() - INTERVAL '10 minutes'
GROUP BY event_id
ORDER BY failed_holds DESC;
\`\`\`

This design balances strong consistency (no double-booking) with user experience (fast holds, clear feedback) and system scalability (high concurrent traffic handling).`,
    keyPoints: [
      'Pessimistic locking (FOR UPDATE) prevents double-booking',
      'Temporary holds with expiration timestamps handle abandoned carts',
      'Background job releases expired holds automatically',
      'Read Committed isolation + row-level locks balance consistency and concurrency',
      'Optimistic path (search) + pessimistic checkout (hold) optimizes UX',
    ],
  },
];
