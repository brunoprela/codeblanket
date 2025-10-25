import { MultipleChoiceQuestion } from '@/lib/types';

export const raceConditionsSynchronizationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rcs-mc-1',
      question: 'What is a race condition?',
      options: [
        'When threads run too fast',
        'When multiple threads access shared data concurrently, causing unpredictable results',
        'When a thread crashes',
        'When code runs sequentially',
      ],
      correctAnswer: 1,
      explanation:
        'Race condition: Multiple threads access shared data concurrently without proper synchronization, causing unpredictable/incorrect results. Example: counter = 0; Thread 1: counter += 1 (read 0, write 1). Thread 2: counter += 1 (read 0, write 1). Expected: 2. Actual: 1 (lost update). Why unpredictable: Depends on thread timing (OS scheduling). Same code produces different results. Hard to debug (non-deterministic). How to prevent: Use locks: with lock: counter += 1. Ensures atomic operations. Only 1 thread in critical section. Common in: Shared counters, bank accounts, caches, any mutable shared state. Not a race condition: Independent threads (no shared state), read-only shared data.',
    },
    {
      id: 'rcs-mc-2',
      question: 'What does a Lock do?',
      options: [
        'Makes code faster',
        'Ensures mutual exclusion - only one thread can hold the lock at a time',
        'Locks the entire program',
        'Prevents threads from running',
      ],
      correctAnswer: 1,
      explanation:
        'Lock provides mutual exclusion: only 1 thread holds lock at a time. How it works: Thread calls lock.acquire() (or with lock). If available: acquires lock, enters critical section. If held: blocks until released. Thread releases lock, next waiting thread acquires. Example: lock = Lock(); with lock: counter += 1 # Only 1 thread executes this at a time. Use cases: Protecting shared state (counters, data structures), Database connections, File access. Trade-offs: Protection: Prevents race conditions. Cost: Serializes access (no parallelism in critical section). Overhead: Lock acquire/release takes time. Best practices: Keep critical section small, Use only when necessary, Release locks promptly.',
    },
    {
      id: 'rcs-mc-3',
      question: 'What is the difference between Lock and Semaphore?',
      options: [
        'They are identical',
        'Lock allows 1 thread, Semaphore allows N threads to access simultaneously',
        'Semaphore is faster',
        'Lock is for async, Semaphore is for threading',
      ],
      correctAnswer: 1,
      explanation:
        'Lock vs Semaphore: Lock: Binary (locked/unlocked). Only 1 thread holds at a time. Example: with lock: access_resource(). Semaphore: Counter-based. N threads can hold simultaneously. Example: semaphore = Semaphore(5); with semaphore: access_resource() # Max 5 concurrent. Use cases: Lock: Exclusive access (only 1), shared mutable state. Semaphore: Limited resources, connection pools (N connections), rate limiting (N per second). Example: Database pool with 10 connections: Semaphore(10) allows 10 concurrent queries. Lock would allow only 1 (wasteful). Implementation: Lock is Semaphore(1). Semaphore is generalized Lock.',
    },
    {
      id: 'rcs-mc-4',
      question: 'What is deadlock and how can it occur?',
      options: [
        'When a thread runs forever',
        'When two or more threads wait for each other to release locks, causing permanent blocking',
        'When locks are too slow',
        'When there are too many threads',
      ],
      correctAnswer: 1,
      explanation:
        "Deadlock: Two or more threads permanently block, each waiting for the other to release a lock. Classic scenario: Thread 1: locks A, waits for B. Thread 2: locks B, waits for A. Neither can proceed (circular wait). Conditions for deadlock (all must be true): Mutual exclusion (locks are exclusive), Hold and wait (hold one lock, wait for another), No preemption (can't force release), Circular wait (T1→T2→T1). Prevention: Lock ordering: Always acquire locks in same order (by ID). Timeout: Try acquire with timeout, retry. Lock-free: Use queues instead of locks. Detection: Hard to detect/debug (timing-dependent). Example: sorted_locks = sorted([lock1, lock2], key=id); for lock in sorted_locks: lock.acquire(). This prevents circular wait.",
    },
    {
      id: 'rcs-mc-5',
      question:
        'Why should critical sections protected by locks be kept small?',
      options: [
        'Locks use a lot of memory',
        'Locks serialize execution - large critical sections reduce parallelism and throughput',
        'Small code is always better',
        'Locks expire after a certain time',
      ],
      correctAnswer: 1,
      explanation:
        'Keep critical sections small because locks serialize execution. Large critical section: with lock: read_data(); process_data(); write_data(); transform(); validate(). Only 1 thread runs this entire block (no parallelism). Small critical section: read_data(); process_data(); with lock: write_data() # Only protect shared state. Process/read can run in parallel. Impact: Large: 10 threads wait for 1 to finish entire block (90% idle). Small: 10 threads process in parallel, only serialize writes. Throughput: Large: ~1 task/time. Small: ~10 tasks/time (10× better). Best practice: Identify minimal shared state. Lock only shared state operations. Do computation outside locks. Example: Bad: with lock: result = expensive_computation(). Good: result = expensive_computation(); with lock: shared_state = result.',
    },
  ];
