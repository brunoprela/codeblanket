export const raceConditionsSynchronizationQuiz = [
  {
    id: 'rcs-q-1',
    question:
      'Bank account system: 1000 concurrent threads transfer money between accounts. Without synchronization, balance becomes incorrect. Implement: (1) Thread-safe Account class with deposit/withdraw, (2) Transfer function that locks BOTH accounts (avoid deadlock), (3) Demonstrate race condition without locks, (4) Prove correctness with locks. Why is lock ordering critical?',
    sampleAnswer:
      'Thread-safe bank account: class Account: def __init__(self, balance): self.balance = balance; self.lock = threading.Lock(). def deposit (self, amount): with self.lock: self.balance += amount. def withdraw (self, amount): with self.lock: if self.balance >= amount: self.balance -= amount; return True; return False. def transfer (from_acc, to_acc, amount): # Deadlock prevention: acquire locks in consistent order; first_lock, second_lock = sorted([from_acc, to_acc], key=id); with first_lock.lock: with second_lock.lock: if from_acc.withdraw (amount): to_acc.deposit (amount); return True; return False. Race condition demo (no locks): def transfer_unsafe (from_acc, to_acc, amount): balance = from_acc.balance; # Read; from_acc.balance -= amount; # Interleaving here causes inconsistency; to_acc.balance += amount. Result: Total balance changes (money created/destroyed). With locks: Total balance always constant. Why lock ordering critical: Without ordering: Thread 1: locks A, waits for B. Thread 2: locks B, waits for A → Deadlock. With ordering: All threads acquire locks in same order (by ID) → No circular wait → No deadlock. Proof: Run 1000 transfers, assert total_balance == initial_balance.',
    keyPoints: [
      'Race condition: Concurrent read-modify-write without locks causes lost updates',
      'Thread-safe Account: Each account has lock, deposit/withdraw protected',
      'Transfer: Lock both accounts, use sorted (by ID) for consistent order (prevent deadlock)',
      'Lock ordering: Prevents circular wait (Thread 1: A→B, Thread 2: B→A deadlock)',
      'Proof: With locks, total balance constant across all transfers',
    ],
  },
  {
    id: 'rcs-q-2',
    question:
      'Compare Lock vs Semaphore vs Event for: (1) Database connection pool (max 10 connections), (2) Ensuring only 1 writer at a time (multiple readers OK), (3) Coordinating start of multiple workers. Implement each pattern. When to use which primitive?',
    sampleAnswer:
      'Database connection pool (Semaphore): class DBPool: def __init__(self): self.semaphore = Semaphore(10); self.connections = [create_conn() for _ in range(10)]. async def get_connection (self): await self.semaphore.acquire(); return connection. async def release (self, conn): self.semaphore.release(). Why: Need N=10 concurrent accesses (not just 1). Single writer, multiple readers (RLock + counter): class RWLock: def __init__(self): self.readers = 0; self.writer_lock = Lock(); self.reader_lock = Lock(). def acquire_read (self): with self.reader_lock: self.readers += 1; if self.readers == 1: self.writer_lock.acquire(). def release_read (self): with self.reader_lock: self.readers -= 1; if self.readers == 0: self.writer_lock.release(). def acquire_write (self): self.writer_lock.acquire(). Why: Writers mutually exclusive. Readers concurrent (only block writer). Coordinating worker start (Event): event = Event(); def worker(): event.wait(); # All workers wait; do_work(). coordinator(): setup(); event.set(); # Start all workers simultaneously. Why: Multiple threads wait for single signal. All wake up together. When to use: Lock: Mutual exclusion (only 1 at a time). Semaphore: Allow N concurrent accesses. Event: Signal/coordinate multiple threads. RWLock: Multiple readers, single writer.',
    keyPoints: [
      'Semaphore(N): Allow N concurrent accesses, use for connection pools (max N connections)',
      'RWLock pattern: Multiple readers concurrent, writers exclusive, complex but efficient',
      'Event: Signal multiple threads, all wait until set(), use for coordination',
      'Lock: Basic mutual exclusion (1 at a time), simplest but most restrictive',
      'Choice depends on access pattern: 1-at-a-time (Lock), N-at-a-time (Semaphore), signal (Event)',
    ],
  },
  {
    id: 'rcs-q-3',
    question:
      'Deadlock scenario: Thread 1 locks A then B. Thread 2 locks B then A. Explain why deadlock occurs. Implement 3 solutions: (1) Lock ordering, (2) Timeout with retry, (3) Lock-free using queue. Compare trade-offs.',
    sampleAnswer:
      'Why deadlock: Thread 1: acquires A, waits for B. Thread 2: acquires B, waits for A. Circular wait: T1→B, T2→A (neither releases) → Deadlock. Solution 1 - Lock ordering: Acquire locks in consistent order (by ID). def transfer (acc1, acc2): locks = sorted([acc1, acc2], key=id); with locks[0].lock: with locks[1].lock: transfer(). Trade-off: No deadlock guaranteed. Requires global ordering knowledge. Solution 2 - Timeout with retry: Try acquire with timeout, retry if fails. def transfer_with_timeout (acc1, acc2): while True: if acc1.lock.acquire (timeout=1): if acc2.lock.acquire (timeout=1): transfer(); break; acc2.lock.release(); else: acc1.lock.release(); time.sleep (random.uniform(0, 0.1)). Trade-off: Eventually succeeds (livelock avoided with backoff). Can be slow (many retries). Not deterministic. Solution 3 - Lock-free with queue: All transfers go through single queue (no locks needed). queue = Queue(); def worker(): while True: transfer_req = queue.get(); execute_transfer (transfer_req). Trade-off: No deadlock possible. Simpler code. Sequential processing (lower throughput). Best: Lock ordering (deterministic, no deadlock, good performance).',
    keyPoints: [
      "Deadlock cause: Circular wait (T1 waits for T2's lock, T2 waits for T1's lock)",
      'Lock ordering: Acquire in consistent order (by ID), prevents circular wait, best solution',
      'Timeout retry: Acquire with timeout, release and retry, eventual success but non-deterministic',
      'Lock-free queue: Single worker thread, no locks needed, simple but sequential',
      'Trade-offs: Lock ordering (best), Timeout (simple but slow), Queue (no parallelism)',
    ],
  },
];
