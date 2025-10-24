/**
 * Distributed Transactions Section
 */

export const distributedTransactionsSection = {
  id: 'distributed-transactions',
  title: 'Distributed Transactions',
  content: `Distributed transactions enable atomic operations across multiple nodes, databases, or services, ensuring data consistency in distributed systems despite the inherent complexity and performance trade-offs.

## Overview

**Distributed transaction** = ACID transaction spanning multiple nodes

**Challenge**: Maintain ACID properties across network boundaries

**ACID reminder**:
- **Atomicity**: All or nothing
- **Consistency**: Valid state before and after
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data persists

**Why difficult in distributed systems?**
- Network failures
- Node failures
- Partial failures
- Network partitions
- Latency

---

## Two-Phase Commit (2PC)

### Most Common Distributed Transaction Protocol

**Participants**:
- **Coordinator**: Orchestrates the transaction
- **Participants**: Nodes involved in transaction

### Phase 1: Prepare (Voting)

\`\`\`
Coordinator                 Participant A          Participant B
     |                            |                      |
     |-- PREPARE ---------------->|                      |
     |-- PREPARE -------------------------------->      |
     |                            |                      |
     |<-- VOTE-COMMIT-------------|                      |
     |<-- VOTE-COMMIT ----------------------------|
     |                            |                      |
\`\`\`

**Prepare phase**:
1. Coordinator sends PREPARE to all participants
2. Each participant:
   - Executes transaction locally (but doesn't commit)
   - Writes to undo log and redo log
   - Locks resources
   - Responds with VOTE-COMMIT or VOTE-ABORT

### Phase 2: Commit (Decision)

**If all voted COMMIT**:
\`\`\`
Coordinator                 Participant A          Participant B
     |                            |                      |
     |-- COMMIT ------------------>|                      |
     |-- COMMIT ------------------------------------>      |
     |                            |                      |
     | (Participants commit)      |                      |
     |                            |                      |
     |<-- ACK ---------------------|                      |
     |<-- ACK ----------------------------------------|
     |                            |                      |
\`\`\`

**If any voted ABORT**:
\`\`\`
Coordinator                 Participant A          Participant B
     |                            |                      |
     |-- ABORT ------------------->|                      |
     |-- ABORT ------------------------------------->      |
     |                            |                      |
     | (Participants rollback)    |                      |
     |                            |                      |
     |<-- ACK ---------------------|                      |
     |<-- ACK ----------------------------------------|
\`\`\`

**Commit phase**:
1. Coordinator decides: COMMIT (if all YES) or ABORT (if any NO)
2. Coordinator sends decision to all participants
3. Participants commit or rollback
4. Participants release locks
5. Participants ACK to coordinator

---

## 2PC Problems

### Blocking Protocol

**Problem**: Coordinator fails after PREPARE, before COMMIT/ABORT

\`\`\`
Participant A (voted COMMIT)
    ↓
Holding locks, waiting for decision
    ↓
Coordinator crashes (decision unknown!)
    ↓
Participant A BLOCKED (can't commit or abort)
    ↓
Locks held indefinitely (blocks other transactions)
\`\`\`

**Impact**:
- Resources locked
- Other transactions blocked
- System unavailable

**Solution**: Timeout + manual intervention (ugly!)

### Performance Issues

**Latency**: Multiple round trips (prepare + commit)
\`\`\`
2 phases × N participants × network RTT
Example: 2 × 3 × 50ms = 300ms minimum
\`\`\`

**Locks held**: Long duration (entire 2PC process)

**Availability**: Any participant unavailable = transaction fails

### Not Partition Tolerant

**Network partition**:
\`\`\`
Coordinator     |  PARTITION  |     Participant B
Participant A   |             |
    ↓
Coordinator can't reach Participant B
    ↓
Transaction must abort (even if B is healthy!)
\`\`\`

---

## Three-Phase Commit (3PC)

### Non-Blocking Alternative to 2PC

**Adds "CanCommit" phase** to avoid blocking

### Phase 1: CanCommit

\`\`\`
Coordinator asks: "Can you commit?"
Participants check: Do I have resources?
Participants respond: YES or NO
(No locks yet!)
\`\`\`

### Phase 2: PreCommit

\`\`\`
If all YES:
  Coordinator: PRECOMMIT
  Participants: Prepare and lock resources
  Participants: ACK
Else:
  Coordinator: ABORT
\`\`\`

### Phase 3: DoCommit

\`\`\`
Coordinator: COMMIT
Participants: Commit and release locks
Participants: ACK
\`\`\`

**Key difference**: Timeout behavior

**2PC**: If coordinator fails after PREPARE, participants blocked
**3PC**: If coordinator fails after PRECOMMIT, participants can commit after timeout (assume consensus)

**Still has problems**:
- More phases = higher latency
- Not safe under network partitions
- Rarely used in practice

---

## Paxos

### Consensus Algorithm

**Purpose**: Agree on single value despite failures

**Participants**:
- **Proposers**: Propose values
- **Acceptors**: Vote on proposals
- **Learners**: Learn chosen value

### Simplified Flow

**Phase 1: Prepare**
\`\`\`
Proposer generates unique proposal number N
Proposer → Acceptors: PREPARE(N)
Acceptors: If N > highest seen:
  - Promise not to accept proposals < N
  - Return highest accepted proposal (if any)
\`\`\`

**Phase 2: Accept**
\`\`\`
If majority responded:
  Proposer → Acceptors: ACCEPT(N, value)
  Acceptors: If N >= promised:
    - Accept proposal
    - Notify learners
\`\`\`

**Chosen value**: When majority of acceptors accept

**Properties**:
- ✅ Fault-tolerant (tolerates < N/2 failures)
- ✅ Non-blocking (makes progress if majority available)
- ❌ Complex to implement correctly
- ❌ Poor performance (many round trips)

**Use cases**:
- Chubby (Google's lock service)
- ZooKeeper (uses Zab, similar to Paxos)
- Cassandra (lightweight transactions)

---

## Raft

### Easier-to-Understand Consensus

**Purpose**: Same as Paxos, but simpler

**Roles**:
- **Leader**: Handles all client requests, replicates log
- **Follower**: Passive, replies to leader requests
- **Candidate**: Becomes leader if leader fails

### Leader Election

\`\`\`
Follower timeout (no heartbeat from leader)
    ↓
Follower becomes Candidate
    ↓
Candidate requests votes from other nodes
    ↓
If majority votes YES:
  Candidate becomes Leader
    ↓
Leader sends heartbeats to maintain authority
\`\`\`

### Log Replication

\`\`\`
Client sends command to Leader
    ↓
Leader appends to local log
    ↓
Leader replicates entry to Followers
    ↓
When majority acknowledge:
  Leader commits entry
  Leader applies to state machine
  Leader responds to client
\`\`\`

**Log structure**:
\`\`\`
Index: 1    2    3    4    5
Term:  1    1    1    2    2
Cmd:  x=1  y=2  z=3  x=4  y=5
\`\`\`

**Properties**:
- ✅ Simpler than Paxos
- ✅ Fault-tolerant (tolerates < N/2 failures)
- ✅ Strong consistency
- ✅ Widely used

**Used by**:
- etcd (Kubernetes)
- Consul
- TiKV
- CockroachDB

---

## Saga Pattern

### Long-Running Transactions Without Locks

**Problem**: Traditional distributed transactions:
- Hold locks for long time
- Block other transactions
- Poor availability

**Saga**: Sequence of local transactions + compensating transactions

### Example: E-Commerce Order

**Traditional 2PC**:
\`\`\`
BEGIN TRANSACTION
  Reserve inventory
  Charge payment
  Create shipment
COMMIT TRANSACTION
(Locks held throughout!)
\`\`\`

**Saga**:
\`\`\`
T1: Reserve inventory → Success
T2: Charge payment → Success  
T3: Create shipment → Failure!
    ↓
Compensating transactions (rollback):
C2: Refund payment
C1: Release inventory
\`\`\`

### Saga Coordination

**1. Choreography** (Event-Driven):

\`\`\`
OrderService creates order
    ↓ (publishes OrderCreated event)
InventoryService reserves inventory
    ↓ (publishes InventoryReserved event)
PaymentService charges payment
    ↓ (publishes PaymentCharged event)
ShipmentService creates shipment
    ↓ (or publishes failure event)
[Compensating events published on failure]
\`\`\`

**Pros**:
- ✅ Decoupled services
- ✅ No single point of failure

**Cons**:
- ❌ Hard to understand flow
- ❌ Difficult to debug

**2. Orchestration** (Coordinator):

\`\`\`
OrderOrchestrator
  ↓
1. Call InventoryService.reserve()
2. Call PaymentService.charge()
3. Call ShipmentService.create()
(If any fails, call compensating actions)
\`\`\`

**Pros**:
- ✅ Centralized logic
- ✅ Easy to understand
- ✅ Easy to debug

**Cons**:
- ❌ Orchestrator = single point of failure
- ❌ Orchestrator = coupling point

### Saga Guarantees

**NOT ACID**:
- ✗ No atomicity (intermediate states visible)
- ✗ No isolation (dirty reads possible)
- ✓ Eventual consistency

**Semantic lock**: Application-level lock (e.g., "order processing")

---

## Spanner (Google)

### Globally Distributed Database with Strong Consistency

**Innovation**: TrueTime API

**TrueTime**: Global wall-clock time with bounded uncertainty

\`\`\`
TrueTime.now() returns interval [earliest, latest]
Uncertainty typically < 10 ms
\`\`\`

**How it works**:
- GPS and atomic clocks in every datacenter
- Timestamp with uncertainty interval
- Wait out uncertainty before commit

**External consistency** (stronger than serializability):
\`\`\`
If transaction T1 commits before T2 starts:
  T1's timestamp < T2's timestamp
\`\`\`

**Benefits**:
- ✅ Global strong consistency
- ✅ Distributed transactions
- ✅ SQL support

**Cost**:
- ❌ Requires specialized hardware
- ❌ Higher latency (wait out uncertainty)
- ❌ Google only (Cloud Spanner available as service)

---

## Calvin

### Deterministic Database

**Idea**: If all replicas execute transactions in same order, they reach same state!

**Architecture**:
\`\`\`
1. Sequencer determines transaction order
2. Order replicated (via Paxos/Raft)
3. All replicas execute in that order
   (Deterministic! No coordination needed!)
\`\`\`

**Benefits**:
- ✅ No 2PC needed
- ✅ Lower latency
- ✅ High throughput

**Limitations**:
- Must know read/write sets in advance
- Dependent transactions need special handling

**Used by**: FaunaDB

---

## Percolator (Google)

### Distributed Transactions on BigTable

**Idea**: Use BigTable cells for transaction metadata

**Transaction protocol**:

**1. Write Phase**:
\`\`\`
For each write:
  Write lock (special column)
  Write data with timestamp
  Designate one write as "primary"
\`\`\`

**2. Commit Phase**:
\`\`\`
Commit primary write (remove lock, add commit timestamp)
If primary commit succeeds:
  Commit secondary writes
  Remove all locks
Else:
  Abort (rollback)
\`\`\`

**Benefits**:
- ✅ No central coordinator
- ✅ Multi-row transactions on BigTable
- ✅ ACID guarantees

**Used by**: Google (internally), TiDB

---

## Best Practices

### 1. Avoid Distributed Transactions When Possible

**Alternatives**:
- Denormalization (duplicate data)
- Eventual consistency (accept lag)
- Application-level consistency
- Saga pattern (long-running transactions)

### 2. Use Idempotency

**Make operations idempotent** (safe to retry):

\`\`\`python
# Bad (not idempotent)
UPDATE accounts SET balance = balance + 100 WHERE id = 123

# Good (idempotent with transaction ID)
INSERT INTO transactions (id, account_id, amount, processed)
VALUES ('txn-123', 123, 100, FALSE)
ON CONFLICT DO NOTHING

IF newly inserted:
  UPDATE accounts SET balance = balance + 100 WHERE id = 123
  UPDATE transactions SET processed = TRUE WHERE id = 'txn-123'
\`\`\`

### 3. Use Compensating Transactions

**Saga pattern** instead of distributed locks

### 4. Design for Eventual Consistency

**Accept temporary inconsistency**:
- Most systems don't need immediate consistency
- Eventual consistency is often sufficient
- Much simpler and more available

### 5. Minimize Transaction Scope

**Keep transactions small and fast**:
- Fewer participants = less coordination
- Shorter transactions = fewer locks
- Less chance of failure

---

## Interview Tips

**Explain distributed transactions in 2 minutes**:
"Distributed transactions ensure ACID across multiple nodes. Two-Phase Commit (2PC) is common: prepare phase (all vote) and commit phase (coordinator decides). 2PC is blocking - coordinator failure leaves participants waiting. Three-Phase Commit adds CanCommit to avoid blocking. Paxos and Raft achieve consensus for leader election and log replication. Saga pattern uses local transactions + compensating transactions for long-running workflows. Spanner uses TrueTime for global strong consistency. Best practice: avoid distributed transactions when possible - use eventual consistency, idempotency, or Saga pattern instead."

**Key trade-offs**:
- Consistency vs availability (CAP theorem)
- Latency vs consistency (2PC adds latency)
- Simplicity vs correctness (eventual consistency vs distributed transactions)
- Blocking vs non-blocking (2PC vs 3PC/Saga)

**Common mistakes**:
- ❌ Using distributed transactions for everything (slow, brittle)
- ❌ Not handling coordinator failure
- ❌ Not making operations idempotent
- ❌ Not considering Saga pattern

**Protocols to mention**:
- 2PC (Two-Phase Commit) - most common, blocking
- 3PC (Three-Phase Commit) - non-blocking, rarely used
- Paxos - consensus, complex
- Raft - consensus, simpler
- Saga - long-running, compensating transactions
- Spanner/TrueTime - Google's globally consistent DB

---

## Key Takeaways

🔑 Distributed transactions = ACID across multiple nodes
🔑 Two-Phase Commit (2PC): prepare + commit, but blocking
🔑 Coordinator failure = participants blocked (major problem)
🔑 Three-Phase Commit: non-blocking but rarely used
🔑 Paxos/Raft: consensus algorithms for leader election and replication
🔑 Saga pattern: local transactions + compensating transactions
🔑 Spanner: TrueTime enables global strong consistency
🔑 Best practice: Avoid distributed transactions when possible
🔑 Alternatives: eventual consistency, idempotency, Saga, denormalization
🔑 Trade-off: Consistency vs availability vs latency
`,
};
