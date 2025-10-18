/**
 * Multiple choice questions for Caching section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const cachingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Your application has 10K QPS with a 90% cache hit rate. The cache goes down. What happens to your database?',
    options: [
      'Database receives 1K QPS (same as before)',
      'Database receives 9K QPS (only the cache hits)',
      'Database receives 10K QPS (all traffic)',
      'Database receives 5K QPS (system automatically throttles)',
    ],
    correctAnswer: 2,
    explanation:
      'Database receives 10K QPS (all traffic). Before: 90% cache hit rate meant 9K QPS served by cache, 1K QPS hit database. After cache down: 100% cache miss rate, all 10K QPS hit database. Result: Database load increases 10× (from 1K to 10K QPS). This can overwhelm the database. Solution: (1) Make cache highly available (Redis Sentinel/Cluster). (2) Implement graceful degradation (rate limiting, fallback responses). (3) Ensure database can handle full load temporarily (over-provision).',
  },
  {
    id: 'mc2',
    question:
      'Which caching pattern should you use for a read-heavy workload where data rarely changes and stale data is acceptable for a few minutes?',
    options: [
      'Write-through cache',
      'Write-back cache',
      'Cache-aside with TTL',
      'Write-around cache',
    ],
    correctAnswer: 2,
    explanation:
      'Cache-aside with TTL is perfect for this scenario. Why: (1) Read-heavy: Cache-aside checks cache first (fast reads). (2) Rarely changes: Low cache miss rate. (3) Stale data OK: TTL of a few minutes is acceptable. Implementation: Check cache first, on miss query database and cache result with TTL (e.g., 300 seconds). Write-through: Overkill (slow writes, unnecessary consistency for this scenario). Write-back: For write-heavy workloads. Write-around: For write-once-read-rarely data.',
  },
  {
    id: 'mc3',
    question:
      'Your cache is 100% full. A new item needs to be cached. The eviction policy is LRU. Which item gets evicted?',
    options: [
      'The oldest item (first added to cache)',
      'The item that has been accessed the fewest times',
      "The item that hasn't been accessed for the longest time",
      'A random item',
    ],
    correctAnswer: 2,
    explanation:
      'LRU (Least Recently Used) evicts the item that hasn\'t been accessed for the longest time. Example: Item A last accessed 10 minutes ago, Item B last accessed 2 minutes ago → Evict A. Option 1 (FIFO): Evicts oldest by insertion time (ignores access patterns). Option 2 (LFU): Evicts least frequently used (access count, not recency). Option 4: Random eviction. LRU is most common because it keeps "hot" (recently accessed) data in cache.',
  },
  {
    id: 'mc4',
    question:
      "You update a user's profile in the database. What should you do with the cached version to avoid serving stale data?",
    options: [
      'Leave it in cache (TTL will eventually expire)',
      'Delete the cache entry immediately',
      'Update both database and cache (write-through)',
      "Do nothing, cache consistency doesn't matter",
    ],
    correctAnswer: 1,
    explanation:
      'Delete the cache entry immediately (cache invalidation). This ensures: (1) Next read will fetch updated data from database. (2) No stale data served. (3) Simple to implement. Option 1 (rely on TTL): Serves stale data until TTL expires (unacceptable for profile updates). Option 3 (write-through): Also valid but requires updating cache logic (delete is simpler). Option 4: Wrong - consistency matters for user-facing data. Best practice: On write, invalidate cache. On read, check cache → miss → query DB → populate cache.',
  },
  {
    id: 'mc5',
    question: 'What is the main risk of write-back (write-behind) caching?',
    options: [
      'Slow writes (every write hits database)',
      'Stale data in cache',
      'Data loss if cache crashes before flushing to database',
      'High database load',
    ],
    correctAnswer: 2,
    explanation:
      'Data loss if cache crashes before flushing to database. Write-back flow: Write to cache → return success immediately → asynchronously flush to database later. Risk: If cache crashes before flush, data in cache (not yet in database) is lost. Mitigation: (1) Use persistent cache (Redis AOF). (2) Frequent flushes (every few seconds). (3) Accept risk for non-critical data (view counts, likes). Benefits: Fast writes, reduced database load. Trade-off: Small data loss risk vs performance.',
  },
];
