/**
 * Multiple choice questions for Inter-Agent Memory & State section
 */

export const interagentmemorystateMultipleChoice = [
  {
    id: 'maas-memory-mc-1',
    question:
      'In a centralized state store, 3 agents (A, B, C) share a Redis database. Agent A writes "status=processing", then Agent B immediately reads the status. What is the PRIMARY advantage of this approach over passing messages directly?',
    options: [
      'Redis is faster than direct message passing',
      'All agents can access current state without waiting for messages',
      'Redis automatically handles conflicts when multiple agents write simultaneously',
      'Centralized state stores are easier to implement than message passing',
    ],
    correctAnswer: 1,
    explanation:
      "The key advantage is shared visibility—any agent can query current state at any time without coordinating message passing. Option A is not necessarily true (depends on use case). Option C is incorrect—Redis doesn't automatically resolve conflicts; you need locking or transactions. Option D is subjective and not the main technical advantage.",
  },
  {
    id: 'maas-memory-mc-2',
    question:
      'Two agents simultaneously read "counter=10" from shared state, both increment to 11, and both write "counter=11". What is this concurrency issue called, and what is the solution?',
    options: [
      'Deadlock; use timeouts',
      'Race condition; use optimistic locking or transactions',
      'Starvation; use priority queues',
      'Dirty read; use database isolation levels',
    ],
    correctAnswer: 1,
    explanation:
      'This is a classic race condition (lost update). Solution: Optimistic locking (check version before write) or transactions (read-modify-write as atomic operation). Deadlock is when agents wait for each other infinitely. Starvation is when one agent never gets resources. Dirty read is reading uncommitted data.',
  },
  {
    id: 'maas-memory-mc-3',
    question:
      'An agent stores "user_preferences" in its private memory. Another agent needs this data. Which design pattern is MOST appropriate?',
    options: [
      "The second agent directly accesses the first agent's private memory",
      'The first agent publishes "user_preferences" to shared state when updated',
      'The second agent sends a request message to the first agent to retrieve the data',
      'Store all data in shared state from the beginning to avoid this issue',
    ],
    correctAnswer: 2,
    explanation:
      'Request-response messaging respects encapsulation—Agent 1 controls access to its private data. Option A violates encapsulation (direct memory access). Option B works but requires Agent 1 to predict what others need. Option D eliminates private state benefits (encapsulation, independent evolution).',
  },
  {
    id: 'maas-memory-mc-4',
    question:
      'A workflow stores checkpoints after each agent step. The checkpoint data grows to 500MB after 10 steps. What is the BEST strategy to manage checkpoint storage?',
    options: [
      'Keep all checkpoints indefinitely for complete history',
      'Only keep the most recent checkpoint (save storage)',
      'Keep recent checkpoints (last 3) plus key milestones, compress old data',
      'Store checkpoints in memory instead of disk',
    ],
    correctAnswer: 2,
    explanation:
      "Balanced approach: Keep recent checkpoints for quick rollback, keep milestones for auditing, compress older data. Option A wastes storage. Option B limits rollback options (what if the most recent checkpoint is corrupted?). Option D doesn't scale (memory is limited and volatile).",
  },
  {
    id: 'maas-memory-mc-5',
    question:
      'An agent caches LLM responses in its private memory. After 100 requests, the cache is 50MB. What is a common strategy for cache eviction?',
    options: [
      'FIFO (First In First Out)—evict oldest entries',
      'LRU (Least Recently Used)—evict entries not accessed recently',
      'Random eviction—randomly remove entries',
      'Never evict—keep growing the cache',
    ],
    correctAnswer: 1,
    explanation:
      "LRU is most effective for caches—recently used data is more likely to be needed again. FIFO doesn't consider access patterns (old data might still be popular). Random eviction is inefficient. Option D causes unbounded memory growth.",
  },
];
